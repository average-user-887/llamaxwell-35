// Experimental q2-only immutable copies for a single fixed target model and stream.
// A fresh process owns these copies through teardown. No source allocation is rewritten.
__host__ __device__ static inline int q8pa_src_word(int dst) {
    const int lane = dst < 256 ? (dst % 128) / 4 : dst - 256;
    const int word = dst < 256 ? (dst / 128)*4 + dst % 4 : 8;
    return 9*lane + word;
}
__global__ void q8pa_pack_rows(Layer source, unsigned * packed) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= NROWS_A * (ROW_H / 4)) return;
    const int row = linear / (ROW_H / 4), word = linear % (ROW_H / 4);
    packed[linear] = ((const unsigned *)rowA(source, row))[q8pa_src_word(word)];
}

static const uint8_t * q8pa_lookup(int device, cudaStream_t stream, const Layer & source, const char * name, bool allow_create) {
    struct Entry {
        Layer original;
        uint8_t * packed;
    };
    static std::mutex pack_mutex;
    std::lock_guard<std::mutex> lk(pack_mutex);
    static std::vector<Entry> entries;
    static int owner_device = -1;
    if (entries.empty()) { owner_device = device; }
    GGML_ASSERT(device==owner_device);
    for (const auto & e : entries) {
        if (e.original.qkv==source.qkv) {
            GGML_ASSERT(e.original.gate==source.gate && e.original.alpha==source.alpha && e.original.beta==source.beta);
            return e.packed;
        }
    }
    GGML_ASSERT(allow_create && entries.size()<30);
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
    GGML_ASSERT(capture==cudaStreamCaptureStatusNone);
    constexpr size_t bytes=size_t(NROWS_A)*ROW_H;
    uint8_t * dst=nullptr;
    CUDA_CHECK(cudaMalloc(&dst,bytes));
    cudaEvent_t begin,end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin,stream));
    q8pa_pack_rows<<<(NROWS_A*(ROW_H/4)+255)/256,256,0,stream>>>(source,(unsigned *)dst);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end,stream)); CUDA_CHECK(cudaEventSynchronize(end));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
    CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
    entries.push_back({source,dst});
    fprintf(stderr,"Q8PA_COPY tensor=%s bytes=%zu copy_ms=%.6f resident_bytes=%zu\n",name,bytes,ms,entries.size()*bytes);
    return dst;
}

extern "C" __attribute__((visibility("default"))) void ggml_cuda_q8pa_prepare(
    const ggml_tensor * qkv, const ggml_tensor * gate, const ggml_tensor * alpha, const ggml_tensor * beta) {
    GGML_ASSERT(env_flag("GGML_CUDA_OCCAMY_GDN_T2_Q8A_PACKED"));
    constexpr int device=0;
    ggml_cuda_set_device(device);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
    GGML_ASSERT(prop.major==5 && prop.minor==2 && prop.multiProcessorCount==24);
    GGML_ASSERT(is_q4_mat(qkv,H,NQKV) && is_q4_mat(gate,H,DI) && is_q4_mat(alpha,H,NVH) && is_q4_mat(beta,H,NVH));
    for (const ggml_tensor * t : {qkv,gate,alpha,beta}) {
        GGML_ASSERT(t->data && t->buffer && t->buffer->buft==ggml_backend_cuda_buffer_type(device));
    }
    Layer source{};
    source.qkv=(const uint8_t *)qkv->data; source.gate=(const uint8_t *)gate->data;
    source.alpha=(const uint8_t *)alpha->data; source.beta=(const uint8_t *)beta->data;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    (void)q8pa_lookup(device,stream,source,qkv->name,true);
    CUDA_CHECK(cudaStreamDestroy(stream));
}
