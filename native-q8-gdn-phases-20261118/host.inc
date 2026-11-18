// N70 is a single-stream, one-fixture diagnostic. Its two modes share event
// placement and defer host reads until all 30 participating layers have launched.
static void n70_launch(const char * mode, int device, int grid, cudaStream_t stream, const Params2 & params, const char * tensor) {
    constexpr int count = 30, expected_grid = 72;
    constexpr int words = expected_grid * NW * 6;
    const bool sample = std::strcmp(mode, "clock") == 0;
    GGML_ASSERT(sample || std::strcmp(mode, "control") == 0);
    static int slot = 0, owner_device = -1;
    static cudaStream_t owner_stream;
    static cudaEvent_t begin[count], end[count];
    static unsigned * device_marks = nullptr;
    static std::set<const void *> weights;
    static std::string tensors[count];
    GGML_ASSERT(grid == expected_grid && slot < count);
    if (slot == 0) {
        cudaStreamCaptureStatus capture;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
        owner_device = device;
        owner_stream = stream;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int standard_blocks, diagnostic_blocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&standard_blocks, gdn_fused2<2>, NT, SMEM3_FLOATS * sizeof(float)));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&diagnostic_blocks, gdn_fused2_clock, NT, SMEM3_FLOATS * sizeof(float)));
        GGML_ASSERT(prop.major == 5 && prop.minor == 2 && prop.multiProcessorCount == 24 &&
                    standard_blocks == 3 && diagnostic_blocks == standard_blocks && grid == diagnostic_blocks * prop.multiProcessorCount);
        std::fprintf(stderr, "Q8CLK_BEGIN mode=%s cc=52 grid=%d sm=%d standard_blocks=%d diagnostic_blocks=%d\n",
                     mode, grid, prop.multiProcessorCount, standard_blocks, diagnostic_blocks);
        for (int i = 0; i < count; ++i) {
            CUDA_CHECK(cudaEventCreate(&begin[i]));
            CUDA_CHECK(cudaEventCreate(&end[i]));
        }
        if (sample) {
            CUDA_CHECK(cudaMalloc(&device_marks, size_t(count) * words * sizeof(unsigned)));
        }
    }
    GGML_ASSERT(device == owner_device && stream == owner_stream && weights.insert(params.p[0].L.qkv).second);
    tensors[slot] = tensor;
    CUDA_CHECK(cudaEventRecord(begin[slot], stream));
    if (sample) {
        gdn_fused2_clock<<<grid, NT, SMEM3_FLOATS * sizeof(float), stream>>>(params, device_marks + size_t(slot) * words);
    } else {
        gdn_fused2<2><<<grid, NT, SMEM3_FLOATS * sizeof(float), stream>>>(params);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end[slot], stream));
    if (++slot == count) {
        CUDA_CHECK(cudaEventSynchronize(end[count - 1]));
        for (int i = 0; i < count; ++i) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin[i], end[i]));
            std::fprintf(stderr, "Q8CLK_EVENT mode=%s layer_slot=%d tensor=%s us=%.6f\n", mode, i, tensors[i].c_str(), ms * 1000.0f);
            CUDA_CHECK(cudaEventDestroy(begin[i]));
            CUDA_CHECK(cudaEventDestroy(end[i]));
        }
        if (sample) {
            std::vector<unsigned> host_marks(size_t(count) * words);
            CUDA_CHECK(cudaMemcpy(host_marks.data(), device_marks, host_marks.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(device_marks));
            FILE * out = std::fopen("/out/q8.clock.u32", "wb");
            GGML_ASSERT(out != nullptr);
            GGML_ASSERT(std::fwrite(host_marks.data(), sizeof(unsigned), host_marks.size(), out) == host_marks.size());
            GGML_ASSERT(std::fclose(out) == 0);
        }
        std::fprintf(stderr, "Q8CLK_END mode=%s distinct_t2_layers=%zu\n", mode, weights.size());
    }
}
