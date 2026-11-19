// Standalone synthetic diagnostic, deliberately unreachable from the GGML graph matcher.
namespace store_probe {
template <class T> struct Buffer {
    T * data = nullptr;
    explicit Buffer(size_t n) { CUDA_CHECK(cudaMalloc((void **)&data, n * sizeof(T))); }
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    ~Buffer() { CUDA_CHECK(cudaFree(data)); }
};

struct Fixture {
    Buffer<float> h{2 * H}, norm{H}, cw{4 * NQKV}, alpha{NVH}, beta{NVH}, out_norm{SD};
    Buffer<uint8_t> w{NROWS_A * ROW_H}, out_w{H * ROW_O};
    Buffer<float> source{SSM_N}, conv{CONV_N};
    Buffer<int32_t> row{1};
    Buffer<float> snapshot{SSM_N}, final_state{SSM_N}, conv_out{2 * CONV_N}, residual{2 * H};
    Buffer<Scratch> scratch{2};

    static uint32_t next(uint32_t & seed) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return seed;
    }
    static void fill(float * dst, size_t n, uint32_t & seed, float center, float amp) {
        std::vector<float> data(n);
        for (float & value : data) value = center + amp * float(int(next(seed) % 1025) - 512) / 512.f;
        CUDA_CHECK(cudaMemcpy(dst, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    static void fill_q4(uint8_t * dst, size_t n, uint32_t & seed, bool packed) {
        std::vector<uint8_t> data(n);
        for (size_t i = 0; i < n; i += QB) {
            data[i] = uint8_t(next(seed));
            data[i + 1] = 0x20 + uint8_t(next(seed) & 3);
            for (int j = 2; j < QB; ++j) data[i + j] = uint8_t(next(seed));
        }
        if (packed) {
            std::vector<uint8_t> reordered(n);
            for (size_t offset = 0; offset < n; offset += ROW_O) {
                for (int word = 0; word < ROW_O / 4; ++word) {
                    std::memcpy(reordered.data() + offset + 4 * word, data.data() + offset + 4 * rp_src(word), 4);
                }
            }
            data.swap(reordered);
        }
        CUDA_CHECK(cudaMemcpy(dst, data.data(), n, cudaMemcpyHostToDevice));
    }
    explicit Fixture(uint32_t seed) {
        fill(h.data, 2 * H, seed, 0.f, 0.25f);
        fill(norm.data, H, seed, 1.f, 0.1f);
        fill(cw.data, 4 * NQKV, seed, 0.f, 0.2f);
        fill(alpha.data, NVH, seed, -0.25f, 0.1f);
        fill(beta.data, NVH, seed, 0.2f, 0.1f);
        fill(out_norm.data, SD, seed, 1.f, 0.1f);
        fill(source.data, SSM_N, seed, 0.f, 0.1f);
        fill(conv.data, CONV_N, seed, 0.f, 0.1f);
        fill_q4(w.data, NROWS_A * ROW_H, seed, false);
        fill_q4(out_w.data, H * ROW_O, seed, true);
        CUDA_CHECK(cudaMemset(row.data, 0, sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(scratch.data, 0, 2 * sizeof(Scratch)));
    }
    Params2 params() const {
        const Layer layer{norm.data, cw.data, alpha.data, beta.data, out_norm.data, w.data,
                          w.data + NQKV * ROW_H, w.data + (NQKV + DI) * ROW_H,
                          w.data + (NQKV + DI + NVH) * ROW_H, out_w.data};
        return Params2{{
            {layer, h.data, residual.data, row.data, conv.data, CONV_N, conv_out.data,
             source.data, SSM_N, snapshot.data, scratch.data, 1.e-6f, 1.e-6f, 1.f, 1.e-6f},
            {layer, h.data + H, residual.data + H, row.data, conv.data, CONV_N, conv_out.data + CONV_N,
             source.data, SSM_N, final_state.data, scratch.data + 1, 1.e-6f, 1.e-6f, 1.f, 1.e-6f}
        }};
    }
    void launch(bool skip_snapshot, cudaStream_t stream) const {
        constexpr size_t bytes = SMEM3_FLOATS * sizeof(float);
        if (skip_snapshot) gdn_store_envelope<<<72, NT, bytes, stream>>>(params());
        else gdn_fused2<2><<<72, NT, bytes, stream>>>(params());
        CUDA_CHECK(cudaGetLastError());
    }
};

static std::vector<float> download(const float * src, size_t n) {
    std::vector<float> ret(n);
    CUDA_CHECK(cudaMemcpy(ret.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
    return ret;
}

static bool equal_finite(const std::vector<float> & control, const float * src, const char * label, int fixture) {
    const auto other = download(src, control.size());
    const size_t different = std::inner_product(control.begin(), control.end(), other.begin(), size_t(0),
        std::plus<size_t>{}, [](float a, float b) { return std::memcmp(&a, &b, sizeof(float)) != 0; });
    const size_t finite = std::count_if(other.begin(), other.end(), [](float v) { return std::isfinite(v); });
    std::printf("GDN_COMPARE fixture=%d field=%s words=%zu finite=%zu differing=%zu\n",
                fixture, label, other.size(), finite, different);
    return !different && finite == other.size();
}

static bool occupancy(const void * function, const char * mode) {
    cudaFuncAttributes attr{};
    int blocks = 0, sms = 0, major = 0, minor = 0, device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, function));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, NT, SMEM3_FLOATS * sizeof(float)));
    std::printf("GDN_RESIDENCY mode=%s sm=%d%d sms=%d active=%d grid=72 threads=%d registers=%d local=%zu shared=%zu\n",
                mode, major, minor, sms, blocks, NT, attr.numRegs, attr.localSizeBytes, SMEM3_FLOATS * sizeof(float));
    return major == 5 && minor == 2 && sms == 24 && blocks >= 3 && attr.localSizeBytes == 0;
}
}

extern "C" __attribute__((visibility("default"))) int ggml_cuda_occamy_store_envelope();
extern "C" int ggml_cuda_occamy_store_envelope() {
    using namespace store_probe;
    if (!occupancy((const void *)gdn_fused2<2>, "control") ||
        !occupancy((const void *)gdn_store_envelope, "elided")) return 2;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Fixture fixtures[] = {Fixture(0x91ca8881), Fixture(0x52c083fd), Fixture(0x309f0c15), Fixture(0x4e8071d0)};
    bool valid = true;
    int index = 0;
    for (const auto & f : fixtures) {
        f.launch(false, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto residual = download(f.residual.data, 2 * H);
        const auto conv = download(f.conv_out.data, 2 * CONV_N);
        const auto final_state = download(f.final_state.data, SSM_N);
        const auto snapshot = download(f.snapshot.data, SSM_N);
        constexpr float sentinel = -128.f;
        const std::vector<float> poison(SSM_N, sentinel);
        const size_t written = std::count_if(snapshot.begin(), snapshot.end(), [sentinel](float v) {
            return std::isfinite(v) && v != sentinel;
        });
        CUDA_CHECK(cudaMemcpy(f.snapshot.data, poison.data(), SSM_N * sizeof(float), cudaMemcpyHostToDevice));
        f.launch(true, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        valid &= equal_finite(residual, f.residual.data, "residual_both", index);
        valid &= equal_finite(conv, f.conv_out.data, "convolution_both", index);
        valid &= equal_finite(final_state, f.final_state.data, "final_state", index);
        valid &= equal_finite(poison, f.snapshot.data, "missing_intermediate", index);
        std::printf("GDN_CONTROL_SNAPSHOT fixture=%d finite_changed=%zu expected=%d\n", index, written, SSM_N);
        valid &= written == SSM_N;
        ++index;
    }
    if (valid) {
        cudaEvent_t begin, end;
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
        for (const bool elide : {false, true}) {
            for (int i = 0; i < 16; ++i) fixtures[i % 4].launch(elide, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        constexpr int reps = 128;
        constexpr bool modes[] = {false, true, true, false, true, false, false, true};
        int order = 0;
        for (const bool elide : modes) {
            CUDA_CHECK(cudaEventRecord(begin, stream));
            for (int i = 0; i < reps; ++i) fixtures[i % 4].launch(elide, stream);
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            std::printf("GDN_INTERVAL order=%d mode=%s reps=%d mean_us=%.6f\n", order++,
                        elide ? "elided" : "control", reps, 1000. * ms / reps);
        }
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::puts(valid ? "GDN_ENVELOPE_OK rollback_intermediate_deliberately_missing" : "GDN_ENVELOPE_INVALID");
    std::fflush(stdout);
    return valid ? 0 : 3;
}
