// A2 microbench: moe_v3<true,false> (base, prod LOP3 path) vs moe_v3<true,true> (shexp gate/up in phase-A idle wave).
// Kernels come from occamy-fused.cu (namespace occamy .. before struct DevState), extracted at build time into moe3_kern.inc.
// Synthetic Q4_0 layers (NL distinct layers, ~453 MB each, rotated so no L2 reuse across launches), NV distinct h vectors.
// Output: bitwise compare of every (layer, h) pair and ABAB timing (median per-launch us over rounds).
// Usage: moe3_mb [rounds=12] [launches_per_block=192] [NL=4]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "moe3_kern.inc"
using namespace occamy;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint32_t hsh(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return (uint32_t) x; }
__global__ void fill_q4(uint8_t * p, size_t nblk, uint64_t seed, float sc) {
    for (size_t b = blockIdx.x * (size_t) blockDim.x + threadIdx.x; b < nblk; b += (size_t) gridDim.x * blockDim.x) {
        uint8_t * q = p + b * QB;
        const uint32_t r = hsh(seed * 0x9E3779B97F4A7C15ULL + b);
        const __half d = __float2half(sc * (0.5f + (r & 0xffff) / 65536.0f));
        memcpy(q, &d, 2);
        for (int i = 0; i < 16; i += 4) { const uint32_t v = hsh(seed + b * 7 + i); memcpy(q + 2 + i, &v, 4); }
    }
}
__global__ void fill_f(float * p, size_t n, uint64_t seed, float off, float sc) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t a = hsh(seed * 31 + i * 2), b = hsh(seed * 31 + i * 2 + 1);
        const float u = (a + 0.5f) / 4294967296.0f, v = (b + 0.5f) / 4294967296.0f;   // sum of 2 uniforms ~ triangular
        p[i] = off + sc * (u + v - 1.0f) * 1.7f;
    }
}
static uint8_t * q4(size_t rows, int row_bytes, uint64_t seed, float sc) {
    uint8_t * p; const size_t n = rows * row_bytes; CK(cudaMalloc(&p, n));
    fill_q4<<<1024, 256>>>(p, n / QB, seed, sc); CK(cudaGetLastError()); return p;
}
static float * fl(size_t n, uint64_t seed, float off, float sc) {
    float * p; CK(cudaMalloc(&p, n * sizeof(float))); fill_f<<<256, 256>>>(p, n, seed, off, sc); CK(cudaGetLastError()); return p;
}
int main(int argc, char ** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 12, per = argc > 2 ? atoi(argv[2]) : 192, NL = argc > 3 ? atoi(argv[3]) : 4;
    const int NV = 8;
    int dev = 0; CK(cudaSetDevice(dev)); cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    const size_t smem = SMEM3_FLOATS * sizeof(float);
    int b0 = 0, b1 = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b0, moe_v3<true, false>, NT3, smem));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, moe_v3<true, true>, NT3, smem));
    const int g0 = b0 * pr.multiProcessorCount, g1 = b1 * pr.multiProcessorCount;
    const int Wv = g1 * NW3, nsa = std::min(FFS, (NTA + Wv - 1) / Wv * Wv - NTA);
    printf("device sm_%d%d nsm %d; grid base %d shidle %d (nsa %d); smem %zu B; NL %d NV %d rounds %d per %d\n",
           pr.major, pr.minor, pr.multiProcessorCount, g0, g1, nsa, smem, NL, NV, rounds, per);
    if (g0 <= 0 || g1 <= 0) { printf("FAIL occupancy\n"); return 2; }
    std::vector<Params> PL(NL);
    for (int l = 0; l < NL; ++l) {
        Layer & L = PL[l].L; const uint64_t s = 1000 + 100 * l;
        L.gate_inp = fl((size_t) NE * H, s + 1, 0.0f, 0.03f);
        L.gate_inp_sh = fl(H, s + 2, 0.0f, 0.03f);
        L.gate_exps = q4((size_t) NE * FF, ROW_GU, s + 3, 0.004f);
        L.up_exps   = q4((size_t) NE * FF, ROW_GU, s + 4, 0.004f);
        L.down_exps = q4((size_t) NE * H, ROW_D, s + 5, 0.004f);
        L.gate_sh = q4(FFS, ROW_GU, s + 6, 0.004f);
        L.up_sh   = q4(FFS, ROW_GU, s + 7, 0.004f);
        L.down_sh = q4(H, ROW_D, s + 8, 0.004f);
        PL[l].norm = fl(H, s + 9, 1.0f, 0.2f);
        PL[l].eps = 1e-6f;
    }
    float * hv = fl((size_t) NV * H, 77, 0.0f, 1.0f);
    float * o0, * o1; CK(cudaMalloc(&o0, (size_t) NL * NV * H * 4)); CK(cudaMalloc(&o1, (size_t) NL * NV * H * 4));
    Scratch * sc; CK(cudaMalloc(&sc, sizeof(Scratch))); CK(cudaMemset(sc, 0, sizeof(Scratch)));
    CK(cudaDeviceSynchronize());
    auto launch = [&](bool sha, int i, float * ob) {
        Params P = PL[i % NL]; const int v = (i / NL) % NV;
        P.h = hv + (size_t) v * H; P.s = sc; P.out = ob + ((size_t) (i % NL) * NV + v) * H;
        if (sha) moe_v3<true, true><<<g1, NT3, smem>>>(P); else moe_v3<true, false><<<g0, NT3, smem>>>(P);
    };
    // bitwise: every (layer, h) pair, twice (second pass after the other arm ran, to catch scratch-state leaks)
    unsigned long long diff = 0; float mx = 0;
    std::vector<float> a((size_t) NL * NV * H), b(a.size());
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < NL * NV; ++i) { launch(false, i, o0); launch(true, i, o1); }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), o0, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), o1, b.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < a.size(); ++k) { uint32_t x, y; memcpy(&x, &a[k], 4); memcpy(&y, &b[k], 4); if (x != y) { ++diff; mx = std::max(mx, fabsf(a[k] - b[k])); } }
    }
    size_t nfin = 0; double asum = 0; for (float x : a) { if (std::isfinite(x)) { ++nfin; asum += fabs(x); } }
    printf("BITWISE base vs shidle: diff %llu / %zu floats, max_abs %.3e (finite %zu, mean|out| %.4f)\n", diff, 2 * a.size(), mx, nfin, asum / a.size());
    // timing: ABAB blocks of `per` launches; layer/h rotate
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 64; ++i) { launch(false, i, o0); launch(true, i, o1); }
    std::vector<double> t0, t1;
    for (int r = 0; r < rounds; ++r) {
        for (int arm = 0; arm < 2; ++arm) {
            const bool sha = (r & 1) ? (arm == 0) : (arm == 1);   // ABBA order across rounds
            CK(cudaEventRecord(e0));
            for (int i = 0; i < per; ++i) launch(sha, i, sha ? o1 : o0);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            (sha ? t1 : t0).push_back(1000.0 * ms / per);
        }
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const double m0 = med(t0), m1 = med(t1);
    printf("TIMING per launch us: base median %.2f (min %.2f max %.2f) shidle median %.2f (min %.2f max %.2f) speedup %.4fx\n",
           m0, *std::min_element(t0.begin(), t0.end()), *std::max_element(t0.begin(), t0.end()),
           m1, *std::min_element(t1.begin(), t1.end()), *std::max_element(t1.begin(), t1.end()), m0 / m1);
    printf("ROUNDS base:"); for (double x : t0) printf(" %.2f", x); printf("\nROUNDS shidle:"); for (double x : t1) printf(" %.2f", x); printf("\n");
    printf("%s\n", diff == 0 ? "PASS bitwise" : "FAIL bitwise");
    return diff == 0 ? 0 : 1;
}
