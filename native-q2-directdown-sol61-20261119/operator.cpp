#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#include <numeric>
#include <string>
#include <vector>

namespace {
constexpr int hidden = 2048;
constexpr int experts = 256;
constexpr int ranks = 8;
constexpr int middle = 512;
constexpr int width = 2;
constexpr int graph_size = 256;
constexpr int reps = 32768;
using Ids = std::array<int32_t, ranks * width>;
using Output = std::array<float, hidden * width>;

void require(bool pass, const char * why) {
    if (!pass) {
        std::fprintf(stderr, "N104 ERROR %s\n", why);
        std::exit(1);
    }
}

struct Rng {
    uint32_t seed;
    uint32_t next() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }
    float signed_unit() { return static_cast<int32_t>(next() & 0xffff) * (2.0f / 65535.0f) - 1.0f; }
};

struct WeightData {
    std::vector<float> router = std::vector<float>(hidden * experts);
    std::array<float, hidden> norm;
    std::array<float, hidden> shared_router;

    WeightData() {
        Rng rng{0x567e12ab};
        for (float & w : router) { w = .022f * rng.signed_unit(); }
        for (float & w : norm) { w = 1.0f + .08f * rng.signed_unit(); }
        for (float & w : shared_router) { w = .019f * rng.signed_unit(); }
    }
};

struct Fixture {
    const char * name;
    Output input;
    Ids ids{};
    std::array<int, width*(width-1)/2> pair_overlap{};
    double shortlist_margin = 1.0;

    Fixture(const char * label, const WeightData & weights, float mix) : name(label) {
        Rng rng{0x25ed4117};
        for (float & x : input) { x = rng.signed_unit(); }
        for (int row = 1; row < width; ++row) {
            for (int i = 0; i < hidden; ++i) {
                float & x = input[row * hidden + i];
                x = mix * x + (1.0f - mix) * input[i];
            }
        }
        for (int row = 0; row < width; ++row) {
            std::array<double, experts> logits{};
            std::array<int, experts> order;
            std::iota(order.begin(), order.end(), 0);
            for (int e = 0; e < experts; ++e) {
                for (int i = 0; i < hidden; ++i) {
                    logits[e] += double(input[row * hidden + i]) * weights.norm[i] * weights.router[e * hidden + i];
                }
            }
            std::partial_sort(order.begin(), order.begin() + ranks + 1, order.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });
            std::copy_n(order.begin(), ranks, ids.begin() + row * ranks);
            for (int k = 0; k < ranks; ++k) {
                shortlist_margin = std::min(shortlist_margin, logits[order[k]] - logits[order[k + 1]]);
            }
        }
        for (int a = 0, p = 0; a < width; ++a) {
            for (int b = a + 1; b < width; ++b, ++p) {
                const auto b_begin = ids.begin() + b * ranks;
                for (int k = 0; k < ranks; ++k) {
                    pair_overlap[p] += std::find(b_begin, b_begin + ranks, ids[a * ranks + k]) != b_begin + ranks;
                }
            }
        }
    }
    void report() const {
        require(shortlist_margin > 1e-5, "router fixture has a near tie");
        for(int row=0;row<width;row++) {
            std::fprintf(stderr,"Q2_COHORT FIXTURE_IDS %s row=%d ids=",name,row);
            for(int k=0;k<ranks;k++) std::fprintf(stderr,"%s%d",k?",":"",ids[row*ranks+k]);
            std::fprintf(stderr,"\n");
        }
        std::fprintf(stderr,"Q2_COHORT FIXTURE %s overlap=%d min_margin=%.8g\n",name,pair_overlap[0],shortlist_margin);
    }
};

ggml_context * context() {
    ggml_context * ctx = ggml_init({graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
    require(ctx, "graph context allocation");
    return ctx;
}

void fill_q4(ggml_tensor * tensor, uint32_t seed) {
    require(tensor->type == GGML_TYPE_Q4_0 && ggml_type_size(tensor->type) == 18, "Q4_0 ABI");
    std::vector<uint8_t> tile(tensor->nb[2]);
    Rng rng{seed};
    for (int e = 0; e < tensor->ne[2]; ++e) {
        for (size_t offset = 0; offset < tile.size(); offset += 18) {
            const auto scale = ggml_fp32_to_fp16(.008f + .002f * rng.signed_unit());
            std::memcpy(tile.data() + offset, &scale, sizeof(scale));
            for (int k = 2; k < 18; k += 4) {
                const uint32_t bits = rng.next();
                std::memcpy(tile.data() + offset + k, &bits, sizeof(bits));
            }
        }
        ggml_backend_tensor_set(tensor, tile.data(), size_t(e) * tile.size(), tile.size());
    }
}

struct Weights {
    ggml_context * ctx = context();
    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, experts);
    ggml_tensor * shared_router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, hidden, middle, experts);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, hidden, middle, experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, middle, hidden, experts);
    ggml_tensor * shared_up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hidden, middle);
    ggml_tensor * shared_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hidden, middle);
    ggml_tensor * shared_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, middle, hidden);
    ggml_backend_buffer_t buffer = nullptr;

    void load(ggml_backend_t backend, const WeightData & data) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "weight buffer allocation");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(norm, data.norm.data(), 0, ggml_nbytes(norm));
        ggml_backend_tensor_set(router, data.router.data(), 0, ggml_nbytes(router));
        ggml_backend_tensor_set(shared_router, data.shared_router.data(), 0, ggml_nbytes(shared_router));
        uint32_t seed = 0x1548ad01;
        for (ggml_tensor * t : {up, gate, down, shared_up, shared_gate, shared_down}) {
            fill_q4(t, ++seed);
        }
    }
    ~Weights() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        ggml_free(ctx);
    }
};

enum class Mode { generic, router };

struct Graph {
    ggml_context * ctx = context();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_tensor * output = nullptr;
    ggml_gallocr_t alloc = nullptr;

    ggml_tensor * select(ggml_tensor * probs) {
        return ggml_argsort_top_k(ctx, probs, ranks);
    }

    ggml_tensor * block(const Weights & w, ggml_tensor * h, const int nt) {
        auto * xn = ggml_mul(ctx, ggml_rms_norm(ctx, h, 1e-6f), w.norm);
        auto * probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, w.router, xn));
        auto * ids = select(probs);
        auto * p = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, experts, nt), ids);
        p = ggml_reshape_2d(ctx, p, ranks, nt);
        auto * norm = ggml_clamp(ctx, ggml_sum_rows(ctx, p), 6.103515625e-5f, INFINITY);
        p = ggml_reshape_3d(ctx, ggml_div(ctx, p, norm), 1, ranks, nt);
        auto * x = ggml_reshape_3d(ctx, xn, hidden, 1, nt);
        auto * up = ggml_mul_mat_id(ctx, w.up, x, ids);
        auto * gate = ggml_mul_mat_id(ctx, w.gate, x, ids);
        auto * down = ggml_mul_mat_id(ctx, w.down, ggml_swiglu_split(ctx, gate, up), ids);
        auto * weighted = ggml_mul(ctx, down, p);
        ggml_tensor * routed = nullptr;
        for (int r = 0; r < ranks; ++r) {
            auto * slice = ggml_view_2d(ctx, weighted, hidden, nt, weighted->nb[2], r * weighted->nb[1]);
            routed = routed ? ggml_add(ctx, routed, slice) : slice;
        }
        auto * su = ggml_mul_mat(ctx, w.shared_up, xn);
        auto * sg = ggml_mul_mat(ctx, w.shared_gate, xn);
        auto * sd = ggml_mul_mat(ctx, w.shared_down, ggml_swiglu_split(ctx, sg, su));
        auto * sw = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w.shared_router, xn));
        return ggml_add(ctx, ggml_add(ctx, routed, ggml_mul(ctx, sd, sw)), h);
    }

    Graph(const Weights & weights, Mode mode) {
        ggml_set_input(input);
        // The same seeded rows are evaluated repeatedly during timing.
        // Pin their allocation so GGML cannot reuse it for the residual.
        ggml_set_output(input);
        if (mode == Mode::router) {
            auto * xn = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-6f), weights.norm);
            output = select(ggml_soft_max(ctx, ggml_mul_mat(ctx, weights.router, xn)));
        } else {
            output = block(weights, input, width);
            ggml_set_name(output, "cohort_native_n2split");
        }
        ggml_build_forward_expand(graph, output);
        require(ggml_nelements(output) == (mode == Mode::router ? ranks * width : hidden * width), "graph output shape");
        int projections = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_MUL_MAT_ID) {
                ++projections;
                require(t->src[2]->ne[0] == ranks && t->src[2]->ne[1] == width, "expert rank/width shape");
            }
        }
        require(projections == (mode == Mode::router ? 0 : 3), "graph projection count");
    }

    void allocate(ggml_backend_t backend) {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(alloc && ggml_gallocr_alloc_graph(alloc, graph), "graph buffer allocation");
    }
    void use(const Fixture & fixture) {
        ggml_backend_tensor_set(input, fixture.input.data(), 0, sizeof(fixture.input));
    }
    void run(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA graph execution");
    }
    ~Graph() {
        if (alloc) { ggml_gallocr_free(alloc); }
        ggml_free(ctx);
    }
};

void check_mappings() {
    std::ifstream maps("/proc/self/maps");
    require(maps.good(), "runtime mapping inventory unavailable");
    const std::array<std::string, 3> expected = {
        "libggml.so.0.26.0", "libggml-base.so.0.26.0", "libggml-cuda.so"};
    std::array<bool, 3> found{};
    for (std::string line; std::getline(maps, line); ) {
        const auto pos = line.find('/');
        if (pos == std::string::npos) { continue; }
        const auto path = line.substr(pos);
        for (size_t i = 0; i < expected.size(); ++i) {
            if (line.find("/" + expected[i]) == std::string::npos) { continue; }
            require(path == "/runtime/" + expected[i], "unexpected ggml library mapping");
            found[i] = true;
        }
    }
    require(std::all_of(found.begin(), found.end(), [](bool value) { return value; }), "missing ggml library mapping");
    for (const auto & lib : expected) { std::fprintf(stderr, "N104 MAP /runtime/%s\n", lib.c_str()); }
}

using Count = unsigned long long (*)(unsigned);
using ReadIds = void (*)(int *);
Count count=nullptr;
using GateUpCount=unsigned long long (*)();
GateUpCount direct_down_count=nullptr;
ReadIds read_ids=nullptr;
using Counters=std::array<unsigned long long,5>;
Counters snapshot() {return {count(0),count(1),count(2),count(3),direct_down_count()};}
void checked_delta(Counters before, Counters after, int runs, bool on) {
    const unsigned n=unsigned(runs);
    const Counters expected={0,0,n,n,on?n:0u};
    for (size_t i=0;i<before.size();++i)
        require(after[i]-before[i]==expected[i],"physical row, pair, acknowledgment or direct down dispatch mismatch");
}
void load_counter() {
    void * h=dlopen("/runtime/libggml-cuda.so",RTLD_NOLOAD|RTLD_NOW);
    require(h,"candidate DSO not mapped");
    count=reinterpret_cast<Count>(dlsym(h,"ggml_cuda_occamy_q2_counter"));
    read_ids=reinterpret_cast<ReadIds>(dlsym(h,"ggml_cuda_occamy_q2_router"));
    direct_down_count=reinterpret_cast<GateUpCount>(dlsym(h,"ggml_cuda_occamy_q2_direct_down_count"));
    Dl_info info{};
    require(direct_down_count && dladdr(reinterpret_cast<void*>(direct_down_count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected direct down counter provider");
    require(count && read_ids && dladdr(reinterpret_cast<void*>(count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected counter provider");
    char exe[256]{};
    require(readlink("/proc/self/exe",exe,sizeof(exe)-1)>0 && std::strcmp(exe,"/operator")==0,"unexpected operator executable");
    require(snapshot()==Counters{},"new process counters dirty");
    std::fprintf(stderr,"Q2_COHORT MAP /operator /runtime/libggml-cuda.so\n");
    dlclose(h);
}
void set_mode(bool on,bool diag=false) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q2_COHORTS","1",1)==0 &&
            setenv("GGML_CUDA_OCCAMY_MOE_Q2_DIRECT_DOWN",on?"1":"0",1)==0 &&
            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS",diag?"1":"0",1)==0,"cannot select typed mode");
}
void compare(ggml_backend_t cuda,Graph & graph,Graph & router,const Fixture & f) {
    set_mode(false);
    router.use(f);
    router.run(cuda);
    Ids generic_ids{};
    for(int r=0;r<width;r++) ggml_backend_tensor_get(router.output,generic_ids.data()+r*ranks,r*router.output->nb[1],ranks*sizeof(int32_t));
    require(generic_ids==f.ids,"generic ordered expert choices differ from CPU");
    Output base{},trial{},repeat{};
    auto run=[&](bool on,Output & dst){
        set_mode(on,true);graph.use(f);
        const auto before=snapshot();graph.run(cuda);checked_delta(before,snapshot(),1,on);
        ggml_backend_tensor_get(graph.output,dst.data(),0,sizeof(dst));
        {
            Ids ids{};read_ids(ids.data());require(ids==f.ids,"physical Q2 candidate expert choices differ from CPU");
            for(int r=0;r<width;r++) {
                std::fprintf(stderr,"Q2_PAIR ROUTES %s mode=%s row=%d ids=",f.name,on?"directdown":"cohorts",r);
                for(int k=0;k<ranks;k++) std::fprintf(stderr,"%s%d",k?",":"",ids[r*ranks+k]);
                std::fprintf(stderr,"\n");
            }
        }
    };
    run(false,base);run(true,trial);run(true,repeat);
    for(int row=0;row<width;row++) {
        int different=0,repeat_diff=0,finite=0;double peak=0.;
        for(int i=row*hidden;i<(row+1)*hidden;i++) {
            finite+=std::isfinite(base[i]) && std::isfinite(trial[i]) && std::isfinite(repeat[i]);
            different+=std::memcmp(&base[i],&trial[i],sizeof(float))!=0;
            repeat_diff+=std::memcmp(&trial[i],&repeat[i],sizeof(float))!=0;
            peak=std::max(peak,std::abs(double(base[i])-trial[i]));
        }
        std::fprintf(stderr,"Q2_COHORT ROW %s row=%d finite=%d different=%d repeat_different=%d max_abs=%.10g\n",
                     f.name,row,finite,different,repeat_diff,peak);
        require(finite==hidden && different==0 && repeat_diff==0,"nonfinite or nonidentical row");
    }
    set_mode(false);
}
void perf(ggml_backend_t cuda,Graph & graph,const Fixture & f) {
    auto batch=[&](bool on,int n) {
        set_mode(on);graph.use(f);const auto before=snapshot();
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<n;i++) graph.run(cuda);
        const auto end=std::chrono::steady_clock::now();
        auto after=snapshot();checked_delta(before,after,n,on);
        Output v{};ggml_backend_tensor_get(graph.input,v.data(),0,sizeof(v));
        require(std::memcmp(v.data(),f.input.data(),sizeof(v))==0,"measured graph mutated input");
        const double us=std::chrono::duration<double,std::micro>(end-start).count()/n;
        std::fprintf(stderr,"Q2_PAIR TIME %s mode=%s reps=%d row0=%llu row1=%llu pair=%llu skip=%llu directdown=%llu us=%.6f\n",
                     f.name,on?"directdown":"cohorts",n,after[0]-before[0],after[1]-before[1],after[2]-before[2],after[3]-before[3],after[4]-before[4],us);
    };
    batch(false,8);batch(true,8);
    const bool first=std::strcmp(f.name,"partial")==0;
    batch(first,reps);batch(!first,reps);batch(!first,reps);batch(first,reps);
    set_mode(false);
}
} // namespace

int main(int argc, char ** argv) {
    require(argc == 2, "expected --fixture, --check, or --perf");
    const bool offline = std::strcmp(argv[1], "--fixture") == 0;
    const bool diagnostic = std::strcmp(argv[1], "--check") == 0;
    require(offline || diagnostic || std::strcmp(argv[1], "--perf") == 0, "unexpected test mode");
    const WeightData data;
    const std::array<Fixture, 2> fixtures = {{{"independent", data, 1.0f}, {"partial", data, .34f}}};
    for (const auto & f : fixtures) { f.report(); }
    const auto & independent = fixtures[0].pair_overlap;
    const auto & partial = fixtures[1].pair_overlap;
    require(*std::max_element(independent.begin(), independent.end()) <= 2, "independent rows overlap too much");
    require(*std::min_element(partial.begin(), partial.end()) >= 2 &&
            *std::max_element(partial.begin(), partial.end()) <= 7, "partial rows outside realistic overlap");
    if(!offline) {
        require(setenv("GGML_CUDA_OCCAMY_MOE_Q2_COUNT","1",1)==0,"enable physical V3 counts");
        require(getenv("GGML_CUDA_OCCAMY_N2SPLIT") && std::strcmp(getenv("GGML_CUDA_OCCAMY_N2SPLIT"),"1")==0,"original N2SPLIT disabled");
    }
    ggml_backend_t cuda = nullptr;
    {
        Weights weights;
        Graph generic(weights, Mode::generic), router(weights, Mode::router);
        if (offline) { return 0; }
        ggml_backend_reg_t reg = ggml_backend_load("/runtime/libggml-cuda.so");
        require(reg && ggml_backend_reg_dev_count(reg) == 1, "expected one CUDA device");
        cuda = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        require(cuda && std::strcmp(ggml_backend_name(cuda), "CUDA0") == 0, "expected CUDA0 backend");
        check_mappings();
        load_counter();
        weights.load(cuda, data);
        generic.allocate(cuda);
        router.allocate(cuda);
        for (const auto & f : fixtures) {
            if (diagnostic) { compare(cuda, generic, router, f); }
            else { perf(cuda, generic, f); }
        }
    }
    const auto done=snapshot();
    std::fprintf(stderr,"Q2_PAIR FINISH row0=%llu row1=%llu pair=%llu skip=%llu directdown=%llu\n",done[0],done[1],done[2],done[3],done[4]);
    ggml_backend_free(cuda);
}
