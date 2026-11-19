#include "sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {
constexpr int vocab = 248320;
constexpr int words = (vocab + 31) / 32;
constexpr int threads = 256;
constexpr int tile = 8 * threads;
constexpr int tiles = (vocab + tile - 1) / tile;
constexpr int frames = 64;
constexpr int samples = 256;
constexpr size_t floats = 2 * vocab;
constexpr size_t raw_bytes = floats * sizeof(float);
constexpr unsigned tie = 1, nonfinite = 2, empty = 4;
using Clock = std::chrono::steady_clock;
#include "mask.inc"

struct Choice { int id; unsigned flags; };
struct Peak { float score; int id; unsigned flags; };
static_assert(sizeof(Choice) == 8);

struct Readback {
    float * raw;
    const Choice * choices;
    float compact_value = 0;
    unsigned raw_reads = 0;
    unsigned compact_reads = 0;
};

[[noreturn]] void fail(const char * reason) {
    std::fprintf(stderr, "%s\n", reason);
    std::exit(1);
}

void check(bool condition, const char * reason) {
    if (!condition) {
        fail(reason);
    }
}

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        fail(cudaGetErrorString(status));
    }
}

template <typename T>
T * device_array(size_t n) {
    T * out = nullptr;
    cuda_check(cudaMalloc(&out, n * sizeof(T)));
    return out;
}

template <typename T>
T * pinned_array(size_t n) {
    T * out = nullptr;
    cuda_check(cudaMallocHost(&out, n * sizeof(T)));
    return out;
}

bool contains(const auto & ids, llama_token id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

common_sampler_ptr sampler(bool ignore_eos) {
    common_params_sampling p;
    p.temp = 0;
    p.seed = 7;
    p.ignore_eos = ignore_eos;
    if (ignore_eos) {
        for (int id : eog_ids) {
            p.logit_bias.push_back({id, -INFINITY});
        }
    }
    auto * model = reinterpret_cast<const llama_model *>(eog_ids.data());
    common_sampler_ptr result(common_sampler_init(model, p));
    check(result != nullptr, "common sampler init failed");
    return result;
}

std::vector<llama_token> ordered_accept(common_sampler * smpl, Readback & rb, const llama_tokens & draft) {
    static const std::vector<int> rows{0, 1};
    return common_sampler_sample_and_accept_n(smpl, reinterpret_cast<llama_context *>(&rb), rows, draft);
}

void verify_tokens(const std::vector<llama_token> & ids, const std::array<int, 2> & expected, bool accepted) {
    check(ids.size() == (accepted ? 2 : 1), "unexpected accept count");
    check(ids[0] == expected[0], "unexpected target row 0 choice");
    if (accepted) {
        check(ids[1] == expected[1], "unexpected target row 1 choice");
    }
}

bool compact_allowed(bool needs_probabilities, bool score_observer) {
    return !needs_probabilities && !score_observer;
}

bool valid(const Choice * out) {
    return out[0].flags == 0 && out[1].flags == 0 && out[0].id >= 0 && out[0].id < vocab &&
           out[1].id >= 0 && out[1].id < vocab;
}

int cpu_check() {
    auto raw_smpl = sampler(true);
    auto compact_smpl = sampler(true);
    const std::array<int, 2> expected{67, 95};
    std::vector<float> scores(floats, 0);
    scores[expected[0]] = 15;
    scores[vocab + expected[1]] = 15;
    scores[eog_ids[0]] = 20;
    scores[vocab + eog_ids[0]] = 20;
    const Choice compact[2]{{expected[0], 0}, {expected[1], 0}};
    Readback raw{scores.data(), nullptr};
    Readback compressed{nullptr, compact};
    const auto begin = Clock::now();
    for (int i = 0; i < 160; ++i) {
        const bool accepted = i % 2 == 0;
        const llama_tokens draft{accepted ? expected[0] : expected[0] + 1};
        verify_tokens(ordered_accept(raw_smpl.get(), raw, draft), expected, accepted);
        verify_tokens(ordered_accept(compact_smpl.get(), compressed, draft), expected, accepted);
        check(common_sampler_last(raw_smpl.get()) == common_sampler_last(compact_smpl.get()), "sampler history drift");
    }
    check(raw.raw_reads == 240 && compressed.compact_reads == 240, "pinned stock sampler did not call the expected getter shims");
    check(raw.compact_reads == 0 && compressed.raw_reads == 0, "wrong output channel");
    check(!compact_allowed(true, false) && !compact_allowed(false, true), "score/probability guard failed");
    check(!valid(std::array<Choice, 2>{{{67, tie}, {95, 0}}}.data()), "tie guard failed");
    check(!valid(std::array<Choice, 2>{{{67, nonfinite}, {95, 0}}}.data()), "nonfinite guard failed");
    check(!valid(std::array<Choice, 2>{{{-1, empty}, {95, 0}}}.data()), "all-masked guard failed");
    auto normal_smpl = sampler(false);
    check(ordered_accept(normal_smpl.get(), raw, llama_tokens{expected[0]}) == llama_tokens{eog_ids[0]}, "EOS opt-out control failed");
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    std::printf("cpu_check=pass raw_reads=240 compact_reads=240 fixtures=160 demo_ms=%.3f\n", elapsed);
    return 0;
}

__device__ Peak merge(Peak a, Peak b) {
    const unsigned invalid = (a.flags | b.flags) & nonfinite;
    if (b.id >= 0 && (a.id < 0 || b.score > a.score)) {
        a = b;
    } else if (a.id >= 0 && b.id >= 0 && b.score == a.score) {
        a.flags |= tie;
    }
    a.flags |= invalid;
    return a;
}

__device__ Peak warp_peak(Peak local) {
    for (int step = 16; step > 0; step /= 2) {
        const Peak other{__shfl_down_sync(0xffffffff, local.score, step),
                         __shfl_down_sync(0xffffffff, local.id, step),
                         __shfl_down_sync(0xffffffff, local.flags, step)};
        if ((threadIdx.x & 31) + step < 32) {
            local = merge(local, other);
        }
    }
    return local;
}

__device__ Peak block_peak(Peak local) {
    __shared__ Peak maxima[threads / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    local = warp_peak(local);
    if (lane == 0) {
        maxima[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        return warp_peak(lane < blockDim.x / 32 ? maxima[lane] : Peak{0, -1, 0});
    }
    return local;
}

__global__ void scan_tiles(const float * input, const uint32_t * masked, Peak * scratch) {
    const float * row = input + blockIdx.y * vocab;
    const int first = blockIdx.x * tile + threadIdx.x;
    Peak local{0, -1, 0};
    for (int i = first; i < min(vocab, int(blockIdx.x + 1) * tile); i += threads) {
        const float score = row[i];
        if (!isfinite(score)) {
            local.flags |= nonfinite;
        } else if (!(__ldg(masked + i / 32) & (1u << (i % 32)))) {
            local = merge(local, Peak{score, i, 0});
        }
    }
    const Peak best = block_peak(local);
    if (threadIdx.x == 0) {
        scratch[blockIdx.y * tiles + blockIdx.x] = best;
    }
}

__global__ void finish_rows(const Peak * scratch, Choice * out) {
    const unsigned i = threadIdx.x;
    const Peak local = i < tiles ? scratch[blockIdx.x * tiles + i] : Peak{0, -1, 0};
    const Peak best = block_peak(local);
    if (i == 0) {
        out[blockIdx.x] = {best.id, best.flags | (best.id < 0 ? empty : 0u)};
    }
}

struct Fixture {
    std::array<int, 2> expected;
};

std::array<uint32_t, words> mask() {
    std::array<uint32_t, words> bits{};
    for (int id : eog_ids) {
        bits[id / 32] |= 1u << (id % 32);
    }
    for (int id : suppression_ids) {
        bits[id / 32] |= 1u << (id % 32);
    }
    return bits;
}

Fixture make_fixture(size_t index, float * out) {
    std::mt19937 generator(47963 + unsigned(index));
    std::uniform_real_distribution<float> scores(-16.0f, 10.0f);
    std::generate(out, out + floats, [&] { return scores(generator); });
    Fixture result{};
    for (int row = 0; row < 2; ++row) {
        const int winner = 80 + (index * 4679 + row * 6871) % (vocab - 100);
        check(!contains(eog_ids, winner) && !contains(suppression_ids, winner), "suppressed fixture winner");
        out[row * vocab + winner] = 17.0f;
        if (index % 2) {
            out[row * vocab + eog_ids[row]] = 23.0f;
        }
        result.expected[row] = winner;
    }
    return result;
}

struct Bench {
    cudaStream_t stream = nullptr;
    float * device = nullptr;
    float * host = nullptr;
    Peak * scratch = nullptr;
    Choice * device_choices = nullptr;
    Choice * host_choices = nullptr;
    uint32_t * device_mask = nullptr;
    std::array<Fixture, frames> fixtures{};
    size_t raw_readbacks = 0, compact_readbacks = 0;

    Bench() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        device = device_array<float>((frames + 1) * floats);
        host = pinned_array<float>(floats);
        scratch = device_array<Peak>(2 * tiles);
        device_choices = device_array<Choice>(2);
        host_choices = pinned_array<Choice>(2);
        device_mask = device_array<uint32_t>(words);
        set_mask(mask());
        for (size_t i = 0; i < frames; ++i) {
            fixtures[i] = make_fixture(i, host);
            upload(i);
        }
    }

    ~Bench() {
        cuda_check(cudaFree(device));
        cuda_check(cudaFreeHost(host));
        cuda_check(cudaFree(scratch));
        cuda_check(cudaFree(device_choices));
        cuda_check(cudaFreeHost(host_choices));
        cuda_check(cudaFree(device_mask));
        cuda_check(cudaStreamDestroy(stream));
    }

    void set_mask(const std::array<uint32_t, words> & bits) {
        cuda_check(cudaMemcpy(device_mask, bits.data(), sizeof(bits), cudaMemcpyHostToDevice));
    }

    void upload(size_t i) {
        cuda_check(cudaMemcpy(device + i * floats, host, raw_bytes, cudaMemcpyHostToDevice));
    }

    Readback copy(size_t i, bool compact) {
        const float * scores = device + i * floats;
        if (compact) {
            scan_tiles<<<dim3(tiles, 2), threads, 0, stream>>>(scores, device_mask, scratch);
            finish_rows<<<2, 128, 0, stream>>>(scratch, device_choices);
            cuda_check(cudaMemcpyAsync(host_choices, device_choices, 2 * sizeof(Choice), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            ++compact_readbacks;
            if (valid(host_choices)) {
                return {nullptr, host_choices};
            }
        }
        cuda_check(cudaMemcpyAsync(host, scores, raw_bytes, cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        ++raw_readbacks;
        return {host, nullptr};
    }
};

void edge_checks(Bench & bench) {
    for (int test = 0; test < 5; ++test) {
        const Fixture fixture = make_fixture(1, bench.host);
        const int winner = fixture.expected[0];
        auto bits = mask();
        const unsigned expected_status = test == 0 ? tie : test == 1 || test == 2 ? nonfinite : test == 3 ? empty : 0;
        if (test == 0) {
            bench.host[16] = bench.host[winner];
        } else if (test == 1) {
            bench.host[39] = std::numeric_limits<float>::quiet_NaN();
        } else if (test == 2) {
            bench.host[eog_ids[0]] = std::numeric_limits<float>::infinity();
        } else if (test == 3) {
            bits.fill(UINT32_MAX);
        } else {
            bench.host[16] = std::nextafter(bench.host[winner], -INFINITY);
        }
        bench.set_mask(bits);
        bench.upload(frames);
        const bool want_fallback = test != 4;
        const auto raw_before = bench.raw_readbacks;
        Readback checked = bench.copy(frames, true);
        check((checked.choices == nullptr) == want_fallback, "wrong fallback election");
        check(bench.raw_readbacks - raw_before == size_t(want_fallback), "wrong fallback copy byte count");
        check((bench.host_choices[0].flags & expected_status) == expected_status, "missing device edge status");
        auto candidate = sampler(true);
        const llama_tokens draft{winner};
        const auto got = ordered_accept(candidate.get(), checked, draft);
        auto baseline = sampler(true);
        Readback full = bench.copy(frames, false);
        check(ordered_accept(baseline.get(), full, draft) == got, "forced fallback did not retain stock outcomes");
    }
    bench.set_mask(mask());
    for (const auto guard : {std::make_pair(true, false), std::make_pair(false, true)}) {
        const auto before = bench.compact_readbacks;
        Readback rb = bench.copy(1, compact_allowed(guard.first, guard.second));
        check(rb.raw && !rb.choices && bench.compact_readbacks == before, "probability or score raw-only guard failed");
    }
    std::printf("edge_checks=7 all_masked_fixture_uses_forced_device_mask=1\n");
}

struct Timing { double total; double copy; double cpu; };
using Groups = std::array<std::vector<Timing>, 4>;

Groups trial(Bench & bench, bool compact) {
    auto smpl = sampler(true);
    Groups result;
    std::array<llama_tokens, 2> draft{{{0}, {0}}};
    const size_t initial_raw = bench.raw_readbacks, initial_compact = bench.compact_readbacks;
    for (int i = -64; i < samples; ++i) {
        const unsigned k = unsigned(i + 64);
        const size_t frame = (k / 2) % frames;
        const auto & expected = bench.fixtures[frame].expected;
        const bool accepted = k % 2 == 0;
        llama_tokens & offered = draft[accepted];
        offered[0] = accepted ? expected[0] : expected[0] + 1;
        const auto start = Clock::now();
        Readback rb = bench.copy(frame, compact);
        const auto copied = Clock::now();
        const auto ids = ordered_accept(smpl.get(), rb, offered);
        const auto done = Clock::now();
        verify_tokens(ids, expected, accepted);
        check(rb.raw_reads == (!compact ? unsigned(ids.size()) : 0u) &&
              rb.compact_reads == (compact ? unsigned(ids.size()) : 0u), "steady route unexpectedly fell back");
        if (i >= 0) {
            const auto us = [](auto value) { return std::chrono::duration<double, std::micro>(value).count(); };
            result[2 * (frame % 2) + !accepted].push_back({us(done - start), us(copied - start), us(done - copied)});
        }
    }
    check(bench.raw_readbacks - initial_raw == (!compact ? samples + 64 : 0) &&
          bench.compact_readbacks - initial_compact == (compact ? samples + 64 : 0), "steady per-channel copy counts differ");
    return result;
}

double median(const std::vector<Timing> & data, double Timing::*field) {
    check(data.size() == samples / 4, "wrong sample count");
    std::vector<double> values;
    for (const auto value : data) {
        values.push_back(value.*field);
    }
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2] + values[values.size() / 2 - 1]) / 2;
}

bool report(const Groups & a, const Groups & b, unsigned mirror) {
    bool passed = true;
    for (unsigned mask_case = 0; mask_case < 2; ++mask_case) {
        std::array<double, 2> saves{};
        for (unsigned reject = 0; reject < 2; ++reject) {
            const auto & na = a[2 * mask_case + reject];
            const auto & nb = b[2 * mask_case + reject];
            const double raw = median(na, &Timing::total), compact = median(nb, &Timing::total);
            saves[reject] = raw - compact;
            std::printf("mirror=%u mask_case=%u rejected=%u samples=%zu raw_us=%.3f compact_us=%.3f raw_copy_us=%.3f compact_copy_us=%.3f raw_cpu_us=%.3f compact_cpu_us=%.3f net_us=%.3f\n",
                        mirror, mask_case, reject, na.size(), raw, compact, median(na, &Timing::copy), median(nb, &Timing::copy),
                        median(na, &Timing::cpu), median(nb, &Timing::cpu), saves[reject]);
        }
        const double weighted = (1429 * saves[0] + 213 * saves[1]) / 1642;
        std::printf("mirror=%u mask_case=%u weighted_net_us=%.3f gate_us=550.000 pass=%u\n", mirror, mask_case, weighted, unsigned(weighted >= 550));
        passed &= weighted >= 550;
    }
    return passed;
}

int main_gpu() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count));
    check(count == 1, "expected one admitted CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, 0));
    check(properties.major == 5 && properties.minor == 2 && std::strstr(properties.name, "M6000"), "wrong CUDA device");
    check(frames * raw_bytes > size_t(properties.l2CacheSize), "GPU fixture working set fits in L2");
    Bench bench;
    edge_checks(bench);
    const auto a1 = trial(bench, false);
    const auto b1 = trial(bench, true);
    const auto b2 = trial(bench, true);
    const auto a2 = trial(bench, false);
    const bool first = report(a1, b1, 1);
    const bool second = report(a2, b2, 2);
    std::printf("technical_pass=1 standalone_gate_pass=%u normal_raw_bytes=%zu normal_compact_bytes=%zu rolling_device_bytes=%zu l2_bytes=%d\n",
                unsigned(first && second), raw_bytes, 2 * sizeof(Choice), frames * raw_bytes, properties.l2CacheSize);
    return 0;
}
} // namespace

// Test-only ABI shims. Pinned libllama-common owns the full sample/accept control
// flow and pinned libllama owns every CPU sampler in its effective chain.
extern "C" {
const llama_model * llama_get_model(const llama_context *) { return reinterpret_cast<const llama_model *>(eog_ids.data()); }
const llama_vocab * llama_model_get_vocab(const llama_model *) { return reinterpret_cast<const llama_vocab *>(eog_ids.data()); }
int32_t llama_vocab_n_tokens(const llama_vocab *) { return vocab; }
bool llama_vocab_is_eog(const llama_vocab *, llama_token id) { return contains(eog_ids, id); }
const llama_token * llama_vocab_get_suppress_tokens(const llama_vocab *, int32_t * n) {
    *n = suppression_ids.size();
    return suppression_ids.data();
}
void llama_synchronize(llama_context *) { /* The harness already synchronized the actual transfer. */ }
float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    auto & rb = *reinterpret_cast<Readback *>(ctx);
    check(rb.raw && !rb.choices && i >= 0 && i < 2, "attempt to read a missing full logit row");
    ++rb.raw_reads;
    return rb.raw + i * vocab;
}
float * llama_get_sampled_probs_ith(llama_context *, int32_t) { return nullptr; }
uint32_t llama_get_sampled_logits_count_ith(llama_context *, int32_t) { return 1; }
float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t) {
    auto & rb = *reinterpret_cast<Readback *>(ctx);
    return rb.choices ? &rb.compact_value : nullptr;
}
llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    auto & rb = *reinterpret_cast<Readback *>(ctx);
    return rb.choices ? const_cast<llama_token *>(&rb.choices[i].id) : nullptr;
}
llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    auto & rb = *reinterpret_cast<Readback *>(ctx);
    if (!rb.choices) {
        return LLAMA_TOKEN_NULL;
    }
    ++rb.compact_reads;
    return rb.choices[i].id;
}
}

int main(int argc, const char ** argv) {
    if (argc != 2) {
        return 2;
    }
    if (std::strcmp(argv[1], "--cpu-check") == 0) {
        return cpu_check();
    }
    if (std::strcmp(argv[1], "--gpu") == 0) {
        return main_gpu();
    }
    return 2;
}
