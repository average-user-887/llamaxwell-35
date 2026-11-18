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
    static thread_local std::vector<Entry> entries;
    static thread_local int owner_device = -1;
    static thread_local cudaStream_t owner_stream = nullptr;
    if (entries.empty()) { owner_device = device; owner_stream = stream; }
    GGML_ASSERT(device==owner_device && stream==owner_stream);
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
