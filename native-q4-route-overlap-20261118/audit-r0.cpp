// Host-only observer of the unchanged CUDA DSO's public MoE dispatch.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include <atomic>
#include "match.inc"
struct ggml_backend_cuda_context;
using match_t = bool (*)(ggml_backend_cuda_context &, const ggml_cgraph *, int, ggml_cuda_occamy_moe_match &);
using launch_t = void (*)(ggml_backend_cuda_context &, const ggml_cuda_occamy_moe_match &, float *, int);
using count_t = unsigned long long (*)();
static uint64_t * words;
static int boundary;
static std::atomic<int> active_scope{0}; // 1 prompt, 2 post-prompt, 0 external

static launch_t launch;
static match_t match;
static count_t count;
static constexpr uint64_t magic = 0x4d4f45345102;
static constexpr char match_sym[] = "_Z33ggml_cuda_occamy_moe_match_windowR25ggml_backend_cuda_contextPK11ggml_cgraphiR26ggml_cuda_occamy_moe_match";
static constexpr char sym[] = "_Z27ggml_cuda_occamy_moe_launchR25ggml_backend_cuda_contextRK26ggml_cuda_occamy_moe_matchPfi";
static constexpr uint64_t route_magic = 0x5152344f564c31;
static constexpr unsigned route_header = 16, layer_words = 29, route_cap = 32, stride_rounds = 16;
static uint64_t * route_words;
static llama_context * target_context;
static thread_local bool route_active;
static thread_local uint64_t seen_layers;
static void fail(const char * msg) {
    fprintf(stderr, "native_moe_audit fatal=%s\n", msg);
    _exit(79);
}
__attribute__((constructor)) static void attach() {
    char exe[1024] = {};
    const auto n = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (n < 0 || strcmp(exe, "/candidate/llama-server") != 0) return;
    const int fd = open("/native-moe-counters", O_RDWR | O_CLOEXEC);
    if (fd < 0) fail("open");
    words = static_cast<uint64_t *>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    if (words == MAP_FAILED || words[0] != 0) fail("map");
    words[0] = magic;
    words[1] = getpid();
    const char * value=getenv("NATIVE_Q4_PROMPT_N");
    boundary=value ? atoi(value) : 0;
    if (boundary!=80000 && boundary!=2000) fail("unexpected_prompt_length");
    const int route_fd = open("/native-route-counters", O_RDWR | O_CLOEXEC);
    if (route_fd < 0) fail("route_open");
    route_words = static_cast<uint64_t *>(mmap(nullptr, 16384, PROT_READ | PROT_WRITE, MAP_SHARED, route_fd, 0));
    close(route_fd);
    if (route_words == MAP_FAILED || route_words[0] != 0) fail("route_map");
    route_words[0] = route_magic;
    route_words[1] = getpid();

}
static void resolve() {
    void * lib = dlopen("/candidate/libggml-cuda.so", RTLD_LAZY | RTLD_NOLOAD | RTLD_LOCAL);
    if (!lib) fail("missing_backend");
    match = reinterpret_cast<match_t>(dlsym(lib,match_sym));
    launch = reinterpret_cast<launch_t>(dlsym(lib,sym));
    count = reinterpret_cast<count_t>(dlsym(lib,"ggml_cuda_occamy_q4_cohort_launch_count"));
    if (!match || !launch || !count) fail("missing_symbols");
    Dl_info origin{};
    if (!dladdr(reinterpret_cast<void *>(launch), &origin) ||
        strcmp(origin.dli_fname,"/candidate/libggml-cuda.so") != 0) fail("wrong_backend");
    if (!words) fail("missing_counters");
    words[2] = 1;
}
static void ensure_resolved() {
    static const bool ready = (resolve(), true);
    (void)ready;
}
static void bump(uint64_t & word) { __atomic_fetch_add(&word, uint64_t{1}, __ATOMIC_RELAXED); }

template<typename Fn>
static Fn next_symbol(const char * name) {
    const auto result = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
    if (!result) fail("route_symbol");
    return result;
}

extern "C" llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    static const auto init = next_symbol<decltype(&llama_init_from_model)>("llama_init_from_model");
    if (route_words && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT) {
        if (params.cb_eval || target_context || boundary != 80000) fail("route_target_context");
        target_context = init(model, params);
        if (target_context) bump(route_words[3]);
        return target_context;
    }
    return init(model, params);
}

static bool route_eval(ggml_tensor * tensor, bool ask, void *) {
    static constexpr char prefix[] = "ffn_moe_topk-";
    if (!tensor || strncmp(tensor->name, prefix, sizeof(prefix)-1) != 0) return !ask;
    const char * lane = tensor->name + sizeof(prefix)-1;
    char * end = nullptr;
    const long layer = strtol(lane, &end, 10);
    if (end == lane || *end || layer < 0 || layer >= 40) return !ask;
    if (ask) return true;
    bump(route_words[7]);
    const uint64_t bit = uint64_t{1} << layer;
    if (seen_layers & bit || tensor->type != GGML_TYPE_I32 || tensor->ne[0] != 8 || tensor->ne[1] != 4 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != sizeof(int32_t) ||
            tensor->nb[1] < 8*sizeof(int32_t) || tensor->nb[1] > 256*sizeof(int32_t) ||
            tensor->nb[1] % sizeof(int32_t) != 0 || !tensor->data || !tensor->buffer) {
        bump(route_words[8]);
        return true;
    }
    static const auto tensor_get = next_symbol<decltype(&ggml_backend_tensor_get)>("ggml_backend_tensor_get");
    static const auto buffer_name = next_symbol<decltype(&ggml_backend_buffer_name)>("ggml_backend_buffer_name");
    if (strncmp(buffer_name(tensor->buffer), "CUDA", 4) != 0) {
        bump(route_words[8]);
        return true;
    }
    unsigned char experts[256] = {};
    for (int row = 0; row < 4; ++row) {
        int32_t ranks[8] = {};
        tensor_get(tensor, ranks, row*tensor->nb[1], sizeof(ranks));
        for (int value : ranks) {
            if (value < 0 || value >= 256 || (experts[value] & (1u << row))) {
                bump(route_words[8]);
                return true;
            }
            experts[value] |= (1u << row);
        }
    }
    uint64_t multiplicity[4] = {};
    unsigned expert_union = 0;
    for (unsigned mask : experts) {
        if (!mask) continue;
        ++expert_union;
        ++multiplicity[__builtin_popcount(mask)-1];
    }
    if (expert_union < 8 || expert_union > 32) {
        bump(route_words[8]);
        return true;
    }
    uint64_t * output = route_words + route_header + layer*layer_words;
    ++output[expert_union-8];
    for (unsigned i = 0; i < 4; ++i) output[25+i] += multiplicity[i];
    seen_layers |= bit;
    return true;
}

extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    static const auto compute = next_symbol<decltype(&ggml_backend_sched_graph_compute_async)>("ggml_backend_sched_graph_compute_async");
    if (!route_active) return compute(sched, graph);
    static const auto set = next_symbol<decltype(&ggml_backend_sched_set_eval_callback)>("ggml_backend_sched_set_eval_callback");
    set(sched, route_eval, nullptr);
    bump(route_words[9]);
    const auto result = compute(sched, graph);
    set(sched, nullptr, nullptr);
    bump(route_words[10]);
    return result;
}
static long target_layer(const ggml_cuda_occamy_moe_match & m) {
    const char * name=m.gate_inp->name;
    char * end=nullptr;
    const long layer=!strncmp(name,"blk.",4) ? strtol(name+4,&end,10) : -1;
    return layer>=0 && layer<40 && end && end!=name+4 && !strcmp(end,".ffn_gate_inp.weight") ? layer : -1;
}
bool ggml_cuda_occamy_moe_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int i, ggml_cuda_occamy_moe_match & m) {
    ensure_resolved();
    if (!match(ctx,graph,i,m)) return false;
    if (m.nt==4 && target_layer(m)<0) {
        bump(words[9]);  // Let the original graph interpreter execute the unmatched window.
        return false;
    }
    return true;
}
void ggml_cuda_occamy_moe_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_moe_match & m, float * out, int row) {
    ensure_resolved();
    if (m.nt != 4) {
        launch(ctx,m,out,row);
        if (m.nt == 1 || m.nt == 2) bump(words[2+m.nt]);
        else bump(words[8]);
        return;
    }
    const long layer=target_layer(m);
    if (layer<0 || out || row) {
        bump(words[8]);
        fail("unexpected_width4_target");
    }
    const auto before = count();
    launch(ctx,m,out,row);
    const auto after = count();
    bump(words[5]);
    __atomic_store_n(&words[6], uint64_t(after), __ATOMIC_RELAXED);
    if (after != before+1) { bump(words[7]); return; }
    bump(words[16+layer]);
    const int scope=active_scope.load(std::memory_order_relaxed);
    if (scope==1 || scope==2) bump(words[(scope==1?64:112)+layer]);
    else bump(words[10]);
}

extern "C" int32_t llama_process(llama_context * ctx,llama_process_type type,llama_batch_ext * batch) {
    using process_t=int32_t (*)(llama_context *,llama_process_type,llama_batch_ext *);
    static const auto original=reinterpret_cast<process_t>(dlsym(RTLD_NEXT,"llama_process"));
    if (!original) fail("missing_original_llama_process");
    if (!words || type!=LLAMA_PROCESS_TYPE_DECODE || !batch || batch->tokens.empty()) return original(ctx,type,batch);
    // Only the main server entry owns prompt/decode classification. The learned head uses a different caller.
    Dl_info caller{};
    const bool target=dladdr(__builtin_return_address(0),&caller) && caller.dli_fname &&
                      strstr(caller.dli_fname,"libllama-server-impl.so");
    int scope=0;
    if (target) {
        const auto first=batch->tokens.front().pos[0], last=batch->tokens.back().pos[0];
        if (first>=0 && first<=last && last<boundary) scope=1;
        else if (first>=boundary && first<=last) scope=2;
        bump(words[160+scope]);
        if (batch->tokens.size()==4) bump(words[164+scope]);
    }
    const int previous=active_scope.exchange(scope,std::memory_order_relaxed);
    const bool tracking = target && ctx == target_context && scope == 2 && batch->tokens.size() == 4 &&
                          __atomic_load_n(&route_words[2], __ATOMIC_ACQUIRE) == 1;
    bool take = false;
    if (tracking) {
        const auto first = batch->tokens.front().pos[0];
        bool contiguous = true;
        for (size_t i = 0; i < 4; ++i) {
            contiguous &= batch->tokens[i].pos[0] == first + static_cast<llama_pos>(i);
        }
        const auto eligible = __atomic_fetch_add(&route_words[4], uint64_t{1}, __ATOMIC_RELAXED);
        if (!contiguous) bump(route_words[8]);
        take = contiguous && eligible % stride_rounds == 0 && route_words[5] < route_cap;
        if (take) bump(route_words[5]);
    }
    const bool old_active = route_active;
    if (take) seen_layers = 0;
    route_active = take;
    const int32_t rc=original(ctx,type,batch);
    route_active = old_active;
    if (take && rc == 0 && seen_layers == ((uint64_t{1} << 40)-1)) bump(route_words[6]);
    active_scope.store(previous,std::memory_order_relaxed);
    return rc;
}
