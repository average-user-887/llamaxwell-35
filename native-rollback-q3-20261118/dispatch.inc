// Native host entry points in the unchanged DSO. Forward the CUDA objects untouched.
// The only by-value argument uses the two-field ABI of gated_delta_net.cuh.
struct ggml_backend_cuda_context;
struct ggml_cuda_occamy_gdn_match;
struct ggml_cuda_gated_delta_net_fused_cache { float * data; int64_t slot_stride; };
static_assert(sizeof(ggml_cuda_gated_delta_net_fused_cache) == 16);
namespace {
std::atomic<bool> count_q3{false};
std::atomic<int> t2_attempts{0}, t2_launched{0}, stock{0}, stock_cache{0}, wrong_shape{0};
template <class F> F cuda_symbol(const char * name) {
    static void * handle = dlopen("/candidate/libggml-cuda.so", RTLD_NOLOAD | RTLD_LAZY);
    if (!handle) {
        std::fprintf(stderr, "N34_FORWARD_ERROR symbol=%s missing_loaded_backend\n", name);
        std::abort();
    }
    auto f = reinterpret_cast<F>(dlsym(handle, name));
    Dl_info origin{};
    if (!f || !dladdr(reinterpret_cast<void *>(f), &origin) || !origin.dli_fname ||
        std::strncmp(origin.dli_fname, "/candidate/", 11) != 0) {
        std::fprintf(stderr, "N34_FORWARD_ERROR symbol=%s wrong_origin\n", name);
        std::abort();
    }
    std::fprintf(stderr, "N34_CUDA_FORWARD symbol=%s path=%s\n", name, origin.dli_fname);
    return f;
}
void count_stock(const ggml_tensor * dst, std::atomic<int> & counter) {
    if (!count_q3.load()) return;
    const auto * v = dst->src[2];
    int32_t snapshots=0;
    std::memcpy(&snapshots,dst->op_params,sizeof(snapshots));
    if (!v || v->ne[0]!=128 || v->ne[1]!=32 || v->ne[2]!=3 || v->ne[3]!=1 || snapshots!=3) ++wrong_shape;
    ++counter;
}
}
bool ggml_cuda_occamy_gdn_t2(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & match) {
    static auto fn = cuda_symbol<decltype(&ggml_cuda_occamy_gdn_t2)>(
        "_Z23ggml_cuda_occamy_gdn_t2R25ggml_backend_cuda_contextRK26ggml_cuda_occamy_gdn_match");
    const bool launched = fn(ctx, match);
    if (count_q3.load()) {
        ++t2_attempts;
        if (launched) ++t2_launched;
    }
    return launched;
}
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static auto fn = cuda_symbol<decltype(&ggml_cuda_op_gated_delta_net)>(
        "_Z28ggml_cuda_op_gated_delta_netR25ggml_backend_cuda_contextP11ggml_tensor");
    fn(ctx, dst);
    count_stock(dst, stock);
}
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                               ggml_cuda_gated_delta_net_fused_cache cache) {
    static auto fn = cuda_symbol<decltype(&ggml_cuda_op_gated_delta_net_fused_cache)>(
        "_Z40ggml_cuda_op_gated_delta_net_fused_cacheR25ggml_backend_cuda_contextP11ggml_tensor37ggml_cuda_gated_delta_net_fused_cache");
    if (count_q3.load() && (!cache.data || cache.slot_stride<128*128*32)) ++wrong_shape;
    fn(ctx, dst, cache);
    count_stock(dst, stock_cache);
}
