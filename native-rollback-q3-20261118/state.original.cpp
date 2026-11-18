#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <unistd.h>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Observe the actual submitted ggml graph at libllama's existing dynamic call.
// Reads tensor metadata only. The original scheduler receives the same arguments.
namespace {
struct GraphRows {
    int calls = 0;
    int input = -1;
    int output = -1;
};
thread_local GraphRows * trace_graph = nullptr;
using SubmitGraph = decltype(&ggml_backend_sched_graph_compute_async);
SubmitGraph original_submit() {
    static auto submit = reinterpret_cast<SubmitGraph>(dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async"));
    return submit;
}
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    if (trace_graph) {
        const auto * input = ggml_graph_get_tensor(graph, "inp_tokens");
        const auto * output = ggml_graph_get_tensor(graph, "result_output");
        ++trace_graph->calls;
        trace_graph->input = input ? static_cast<int>(input->ne[0]) : -1;
        trace_graph->output = output ? static_cast<int>(output->ne[1]) : -1;
    }
    const auto submit = original_submit();
    if (!submit) return GGML_STATUS_FAILED;
    return submit(sched, graph);
}

namespace {

using Tokens = std::vector<llama_token>;
constexpr int context_size = 84224;
constexpr int logical_batch = 2048;
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}


Tokens load_ints(const char * path) {
    std::ifstream file(path);
    require(file.is_open(), "cannot open oracle fixture");
    Tokens ids;
    llama_token value;
    while (file >> value) ids.push_back(value);
    require(file.eof() && !ids.empty(), "malformed or empty fixture");
    return ids;
}

std::atomic<bool> offloaded{false};
void log_callback(ggml_log_level, const char * text, void *) {
    if (std::strstr(text, "offloaded 42/42 layers to GPU")) offloaded.store(true);
    std::fputs(text, stderr);
}

struct Decision {
    llama_token id = -1;
    float top = -std::numeric_limits<float>::infinity();
    float second = -std::numeric_limits<float>::infinity();
};

Decision greedy(const float * scores, const std::vector<unsigned char> & excluded) {
    Decision best;
    for (size_t i = 0; i < excluded.size(); ++i) {
        const float x = scores[i];
        require(std::isfinite(x), "nonfinite target logits");
        if (excluded[i]) continue;
        if (x > best.top) {
            best.second = best.top;
            best.top = x;
            best.id = static_cast<llama_token>(i);
        } else if (x > best.second) {
            best.second = x;
        }
    }
    require(best.id >= 0 && std::isfinite(best.second), "no eligible greedy choice");
    if (best.top == best.second) {
        // The reference samples after top-k=40; its partial sort can reorder equal maxima.
        std::vector<llama_token_data> values(excluded.size());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = {static_cast<llama_token>(i),
                         excluded[i] ? -std::numeric_limits<float>::infinity() : scores[i], 0.0f};
        }
        llama_token_data_array data{values.data(), values.size(), -1, false};
        using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
        Sampler topk(llama_sampler_init_top_k(40), llama_sampler_free);
        llama_sampler_apply(topk.get(), &data);
        best.id = data.data[0].id;
    }
    return best;
}

struct Batch {
    llama_batch b = llama_batch_init(logical_batch, 0, 1);
    ~Batch() { llama_batch_free(b); }
    void fill(const Tokens & ids, int index, int absolute, int count, bool all, bool last) {
        require(count > 0 && count <= logical_batch && index >= 0 && index + count <= static_cast<int>(ids.size()),
                "out-of-bounds token batch");
        b.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            b.token[i] = ids[index+i];
            b.pos[i] = absolute+i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = all || (last && i == count-1);
        }
    }
};


}

#include "dispatch.inc"

namespace {
constexpr size_t partial_size = 65864428;
void write_file(const std::string & file, const void * data, size_t size) {
    std::ofstream out(file, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    out.close();
}
void run(const std::string & mode, const char * model_path, const Tokens & prompt,
         const Tokens & oracle, const Tokens & ends, const std::string & out) {
    const int n_prompt = static_cast<int>(prompt.size());
    require(n_prompt == 2000 && oracle.size() == 192 && ends == Tokens({972,1996,2000}), "wrong fixture");
    require(std::string(getenv("GGML_CUDA_OCCAMY_FA_Q3_R2") ? getenv("GGML_CUDA_OCCAMY_FA_Q3_R2") : "") == "1",
            "N100 q3 profile absent");
    llama_log_set(log_callback, nullptr);
    ggml_backend_load_all_from_path("/candidate");
    llama_backend_init();
    Dl_info llama_origin{};
    require(dladdr(reinterpret_cast<void *>(&llama_decode), &llama_origin) &&
            llama_origin.dli_fname && std::strncmp(llama_origin.dli_fname, "/candidate/", 11) == 0,
            "libllama resolved outside pinned runtime");
    std::fprintf(stderr, "N34_RUNTIME llama=%s\n", llama_origin.dli_fname);
    std::set<std::string> expected;
    for (const char * lib : {"libllama.so","libggml.so","libggml-base.so","libggml-cuda.so"}) {
        char actual[PATH_MAX]{};
        require(realpath((std::string("/candidate/")+lib).c_str(),actual), "cannot resolve library");
        expected.insert(actual);
    }
    std::ifstream maps("/proc/self/maps");
    std::set<std::string> found;
    std::string line;
    while (std::getline(maps,line)) {
        auto slash=line.find('/');
        if(slash==std::string::npos) continue;
        auto path=line.substr(slash), name=path.substr(path.find_last_of('/')+1);
        if(name.rfind("libllama.so",0)==0 || name.rfind("libggml.so",0)==0 ||
           name.rfind("libggml-base.so",0)==0 || name.rfind("libggml-cuda.so",0)==0) found.insert(path);
    }
    require(found==expected && expected.size()==4, "unexpected actual runtime mappings");
    for(const auto & path:found) std::fprintf(stderr,"ROLL3_MAP %s\n",path.c_str());
    const auto cuda = ggml_backend_dev_by_name("CUDA0");
    require(cuda && ggml_backend_dev_type(cuda) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            std::strstr(ggml_backend_dev_description(cuda), "M6000"), "admitted CUDA0 M6000 unavailable");
    ggml_backend_dev_t devices[]{cuda, nullptr};
    llama_model_kv_override overrides[2]{};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    std::strcpy(overrides[0].key, "tokenizer.ggml.pre");
    std::strcpy(overrides[0].val_str, "qwen35");
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 999;
    mp.kv_overrides = overrides;
    mp.progress_callback = [](float,void *){ return true; };
    using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    Model model(llama_model_load_from_file(model_path, mp), llama_model_free);
    require(model && offloaded.load(), "missing actual 42/42 GPU offload proof");
    const auto vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    require(n_vocab == 248320, "unexpected vocabulary");
    for (const auto * group : {&prompt, &oracle}) {
        for (auto id : *group) require(id >= 0 && id < n_vocab, "fixture token out of vocabulary");
    }
    std::vector<unsigned char> excluded(n_vocab);
    int suppressed = 0;
    for (int i = 0; i < n_vocab; ++i) {
        excluded[i] = llama_vocab_is_eog(vocab, i);
        suppressed += excluded[i];
    }
    require(suppressed == 5, "unexpected 2K fixture EOS policy");
    auto cp = llama_context_default_params();
    cp.n_ctx = context_size;
    cp.n_batch = logical_batch;
    cp.n_ubatch = 1024;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 2;
    cp.n_outputs_max = 3;
    cp.n_outputs_max_per_seq = 3;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = cp.op_offload = true;
    cp.kv_unified = false;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
    Context ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(ctx && llama_n_ctx(ctx.get()) == context_size && llama_n_rs_seq(ctx.get()) == 2, "context shape mismatch");
    auto mem = llama_get_memory(ctx.get());
    require(mem && llama_memory_seq_pos_max(mem, 0) == -1, "new sequence has prior state");
    Batch batch;
    int previous = 0;
    for (auto end : ends) {
        require(end > previous && end <= n_prompt, "invalid prefill boundary");
        batch.fill(prompt, previous, previous, end-previous, false, end == n_prompt);
        require(llama_decode(ctx.get(), batch.b) == 0, "prefill decode failed");
        previous = end;
    }
    llama_synchronize(ctx.get());
    require(llama_memory_seq_pos_max(mem, 0) == n_prompt-1, "prefill position mismatch");
    const float * last_logits = llama_get_logits(ctx.get());
    require(last_logits, "missing prefill logits");
    const auto base = out + "/" + mode + ".";
    const auto pos = [&]{ return llama_memory_seq_pos_max(mem, 0); };
    const auto snapshot = [&](const char * stage) {
        llama_synchronize(ctx.get());
        const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const auto n = llama_state_seq_get_size_ext(ctx.get(),0,flags);
        require(n == partial_size, "unexpected recurrent payload size");
        std::vector<uint8_t> data(n);
        require(llama_state_seq_get_data_ext(ctx.get(),data.data(),n,0,flags) == n, "incomplete partial state");
        write_file(base+stage+".state", data.data(), data.size());
        // For this pinned hybrid writer flags=0 emits attention KV, then exactly the same recurrent record.
        const auto total=llama_state_seq_get_size_ext(ctx.get(),0,0);
        require(total>n && total-n<128*1024*1024,"unexpected full hybrid serialization length");
        std::vector<uint8_t> full(total);
        require(llama_state_seq_get_data_ext(ctx.get(),full.data(),total,0,0)==total &&
                std::memcmp(full.data()+total-n,data.data(),n)==0,"full state's recurrent suffix differs");
        write_file(base+stage+".kv",full.data(),total-n);
        std::cout<<"ROLL3_KV stage="<<stage<<" pos="<<pos()<<" bytes="<<total-n<<" full_bytes="<<total<<"\n";
        std::cout << "N34_STATE stage=" << stage << " pos=" << pos() << " bytes=" << n << "\n";
    };
    auto decode = [&](const char * stage, const Tokens & input, bool keep_logits) {
        int at = pos()+1;
        int width = static_cast<int>(input.size());
        batch.fill(input,0,at,width,true,true);
        GraphRows shape;
        trace_graph = &shape;
        count_q3.store(width == 3);
        require(llama_decode(ctx.get(),batch.b) == 0, "decode failed");
        llama_synchronize(ctx.get());
        count_q3.store(false);
        trace_graph = nullptr;
        require(shape.calls == 1 && shape.input == width && shape.output == width && pos() == at+width-1,
                "actual submitted shape or resulting position mismatch");
        const float * logits = llama_get_logits(ctx.get());
        require(logits, "missing logits");
        if (keep_logits) write_file(base+stage+".logits",logits,size_t(width)*n_vocab*sizeof(float));
        std::vector<llama_token> decisions;
        for (int i=0; i<width; ++i) decisions.push_back(greedy(logits+size_t(i)*n_vocab,excluded).id);
        std::cout << "N34_CALL stage=" << stage << " at=" << at << " graphs=" << shape.calls
                  << " in=" << shape.input << " out=" << shape.output << " input0=" << input.front()
                  << " input_last=" << input.back() << " next0=" << decisions.front()
                  << " next_last=" << decisions.back() << "\n";
        return decisions;
    };

    std::cout << "N34_START mode=" << mode << " offload=42/42 rs=" << llama_n_rs_seq(ctx.get())
              << " prompt=" << n_prompt << " prefill_next=" << greedy(last_logits,excluded).id
              << " vocab=" << n_vocab << " excluded=" << suppressed << "\n";
    for (int i=0; i<4; ++i) decode("prefix",{oracle[i]},false);
    snapshot("prefix");
    const auto keep=oracle[4];
    const llama_token wrong=mode=="wide_a"?42:44, tail=mode=="wide_a"?43:45;
    llama_token first_choice;
    if (mode=="serial") {
        first_choice=decode("kept",{keep},true).front();
    } else {
        const auto decisions=decode("badtriple",{keep,wrong,tail},true);
        first_choice=decisions.front();
        snapshot("badtriple");
        require(first_choice!=wrong,"first proposal was not rejected");
        const auto before=pos();
        require(before==2006 && llama_memory_seq_rm(mem,0,before-1,-1),"two-position native rewind refused");
        require(pos()==2004,"incorrect retained position after rewind");
        std::cout<<"ROLL3_REMOVE start="<<before-1<<" after="<<pos()<<" before="<<before
                 <<" wrong="<<wrong<<" tail="<<tail<<" first_choice="<<first_choice<<"\n";
    }
    snapshot("kept");
    const auto chosen=out+"/correction.id";
    if(mode=="wide_a") {
        std::ofstream saved(chosen);
        saved<<first_choice<<"\n";
        require(saved.good(),"failed to write actual wide correction");
    }
    const auto correction=load_ints(chosen.c_str());
    require(correction.size()==1 && correction[0]>=0 && correction[0]<n_vocab,"invalid accepted correction");
    decode("corrected",correction,true);
    snapshot("corrected");
    for(int i=6;i<10;++i) decode("follow",{oracle[i]},false);
    std::cout << "N34_END mode=" << mode << " t2_attempts=" << t2_attempts << " t2_launches=" << t2_launched
              << " stock=" << stock << " stock_cache=" << stock_cache << " wrong_shape=" << wrong_shape
              << " end=" << pos() << " first_choice=" << first_choice << "\n";
}
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            require(original_submit(), "scheduler forwarding unavailable");
            const float scores[] = {1.0,4.0,8.0};
            require(greedy(scores,{0,0,1}).id == 1, "EOS-masked greedy failed");
            require(8+20+30*(98304+2097152+24) == partial_size,"state dimensions inconsistent");
            std::puts("N34_CPU_PASS partial_bytes=65864428 layers=30 q3_inputs=3 q3_outputs=3");
            return 0;
        }
        require(argc==7,"usage: state MODE MODEL PROMPT_IDS ORACLE_IDS PREFILL_ENDS OUTDIR");
        std::string mode(argv[1]);
        require(mode == "serial" || mode == "wide_a" || mode == "wide_b","unknown mode");
        run(mode,argv[2],load_ints(argv[3]),load_ints(argv[4]),load_ints(argv[5]),argv[6]);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr,"N34_ERROR %s\n",e.what());
        return 2;
    }
}
