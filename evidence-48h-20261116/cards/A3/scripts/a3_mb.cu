// A3 microbench: gdn_fused2<A2ROW=1, IL> two-slot decode op (real shapes, synthetic data, in-place state rows as in prod
// N2GDN). IL 0 = A2ROW baseline (per sequence), 1 = card 18, 2 = A3 chain interleave, 3 = A3 + next-item state prefetch.
// Kernels come from the A3 worktree occamy-gdn.cu (namespace occamy_gdn up to struct DevState), extracted at build time.
// GDN state is recurrent (fixed 2 x 2.1 MB per layer), so context depth does not change this op's shape.
// Output: bitwise compare of each IL vs IL 0 after 3 chained launches (out, conv, ssm, scratch), then interleaved timing
// (R rounds x 4 variants x N launches, median us/launch per variant). Usage: a3_mb [rounds=12] [launches=128] [grid=0 -> prod]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>
#include <algorithm>
#include "a3_kern.inc"
using namespace occamy_gdn;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static uint32_t hsh(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
static void *dbytes(size_t n, uint32_t seed, uint8_t mask) {
    std::vector<uint8_t> h(n); for (size_t i = 0; i < n; ++i) h[i] = hsh((uint32_t)i * 2654435761u + seed) & mask;
    void *d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice)); return d;
}
static float *dfl(size_t n, uint32_t seed, float lo, float hi) {
    std::vector<float> h(n); for (size_t i = 0; i < n; ++i) h[i] = lo + (hi - lo) * (hsh((uint32_t)i * 747796405u + seed) / 4294967296.0f);
    float *d; CK(cudaMalloc(&d, n * 4)); CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice)); return d;
}
typedef void (*KF)(const Params2);
int main(int argc, char **argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 12, N = argc > 2 ? atoi(argv[2]) : 128, gopt = argc > 3 ? atoi(argv[3]) : 0;
    KF kf[4] = { gdn_fused2<1, 0>, gdn_fused2<1, 1>, gdn_fused2<1, 2>, gdn_fused2<1, 3> };
    const char *nm[4] = { "IL0 A2ROW base", "IL1 card18", "IL2 A3", "IL3 A3+PF" };
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const size_t SM = SMEM3_FLOATS * sizeof(float);
    int psm[4]; for (int v = 0; v < 4; ++v) { CK(cudaFuncSetAttribute(kf[v], cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SM));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&psm[v], kf[v], NT, SM)); cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kf[v]));
        printf("%s: %d blocks/SM, %d regs, %zu B local, %zu B static smem\n", nm[v], psm[v], fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes); }
    const int g = gopt > 0 ? gopt : psm[0] * pr.multiProcessorCount;
    printf("grid %d (dyn smem %zu B), rounds %d x %d launches\n", g, SM, R, N);
    Layer L;
    L.attn_norm = dfl(H, 1, 0.5f, 1.5f); L.conv_w = dfl(NQKV * 4, 2, -0.5f, 0.5f); L.ssm_a = dfl(NVH, 3, -2.f, -0.1f);
    L.dt_bias = dfl(NVH, 4, -1.f, 1.f); L.ssm_norm = dfl(SD, 5, 0.5f, 1.5f);
    L.qkv = (const uint8_t *)dbytes((size_t)NQKV * ROW_H, 6, 0x1F); L.gate = (const uint8_t *)dbytes((size_t)DI * ROW_H, 7, 0x1F);
    L.alpha = (const uint8_t *)dbytes((size_t)NVH * ROW_H, 8, 0x1F); L.beta = (const uint8_t *)dbytes((size_t)NVH * ROW_H, 9, 0x1F);
    L.out = (const uint8_t *)dbytes((size_t)H * ROW_O, 10, 0x1F);
    float *hbuf = dfl(2 * H, 11, -1.f, 1.f), *obuf; CK(cudaMalloc(&obuf, 2 * H * 4));
    float *conv = dfl(2 * (size_t)CONV_N, 12, -1.f, 1.f), *st = dfl(2 * (size_t)SSM_N, 13, -0.1f, 0.1f);
    int32_t *z; CK(cudaMalloc(&z, 4)); CK(cudaMemset(z, 0, 4));
    Scratch *sc[2]; for (int k = 0; k < 2; ++k) { CK(cudaMalloc(&sc[k], sizeof(Scratch))); CK(cudaMemset(sc[k], 0, sizeof(Scratch))); }
    Params2 PP;
    for (int k = 0; k < 2; ++k) {
        Params &P = PP.p[k]; P.L = L; P.h = hbuf + k * H; P.out = obuf + k * H; P.s_copy = z;
        P.conv_base = conv + (size_t)k * CONV_N; P.conv_stride = CONV_N; P.conv_out = conv + (size_t)k * CONV_N;
        P.ssm_base = st + (size_t)k * SSM_N; P.ssm_stride = SSM_N; P.S_out = st + (size_t)k * SSM_N;
        P.s = sc[k]; P.eps_attn = 1e-6f; P.eps_l2 = 1e-6f / 128; P.scale_l2 = 0.088388f; P.eps_gated = 1e-6f;
    }
    {   // bit-equality, same snapshot for every variant
        const size_t nc = 2 * (size_t)CONV_N, ns = 2 * (size_t)SSM_N, nx = offsetof(Scratch, bar_count);
        std::vector<float> c0(nc), s0v(ns); CK(cudaMemcpy(c0.data(), conv, nc * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(s0v.data(), st, ns * 4, cudaMemcpyDeviceToHost));
        std::vector<char> sc0[2]; for (int k = 0; k < 2; ++k) { sc0[k].resize(sizeof(Scratch)); CK(cudaMemcpy(sc0[k].data(), sc[k], sizeof(Scratch), cudaMemcpyDeviceToHost)); }
        std::vector<float> ro[4], rc[4], rs[4]; std::vector<char> rx[4][2];
        for (int v = 0; v < 4; ++v) {
            CK(cudaMemcpy(conv, c0.data(), nc * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(st, s0v.data(), ns * 4, cudaMemcpyHostToDevice));
            for (int k = 0; k < 2; ++k) CK(cudaMemcpy(sc[k], sc0[k].data(), sizeof(Scratch), cudaMemcpyHostToDevice));
            for (int it = 0; it < 3; ++it) kf[v]<<<g, NT, SM>>>(PP);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            ro[v].resize(2 * H); rc[v].resize(nc); rs[v].resize(ns);
            CK(cudaMemcpy(ro[v].data(), obuf, 2 * H * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rc[v].data(), conv, nc * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rs[v].data(), st, ns * 4, cudaMemcpyDeviceToHost));
            for (int k = 0; k < 2; ++k) { rx[v][k].resize(nx); CK(cudaMemcpy(rx[v][k].data(), sc[k], nx, cudaMemcpyDeviceToHost)); }
        }
        for (int v = 1; v < 4; ++v) {
            size_t d = 0; double mx = 0; for (size_t i = 0; i < ns; ++i) if (memcmp(&rs[0][i], &rs[v][i], 4)) { ++d; mx = std::max(mx, (double) fabsf(rs[0][i] - rs[v][i])); }
            printf("BITWISE %s vs IL0: out %d conv %d ssm %d (diff %zu, maxabs %.3e) scratch %d\n", nm[v], !memcmp(ro[0].data(), ro[v].data(), 2 * H * 4),
                   !memcmp(rc[0].data(), rc[v].data(), nc * 4), !memcmp(rs[0].data(), rs[v].data(), ns * 4), d, mx,
                   !memcmp(rx[0][0].data(), rx[v][0].data(), nx) && !memcmp(rx[0][1].data(), rx[v][1].data(), nx));
        }
    }
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int v = 0; v < 4; ++v) for (int i = 0; i < 16; ++i) kf[v]<<<g, NT, SM>>>(PP);
    CK(cudaDeviceSynchronize());
    std::vector<float> t[4];
    for (int r = 0; r < R; ++r) for (int q = 0; q < 4; ++q) {
        const int v = (r & 1) ? 3 - q : q;   // ABCD DCBA
        CK(cudaEventRecord(e0)); for (int i = 0; i < N; ++i) kf[v]<<<g, NT, SM>>>(PP);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t[v].push_back(ms * 1000.f / N);
    }
    float med[4];
    for (int v = 0; v < 4; ++v) { std::sort(t[v].begin(), t[v].end()); med[v] = t[v][t[v].size() / 2]; }
    for (int v = 0; v < 4; ++v) printf("TIME %s: median %.2f us/launch (min %.2f max %.2f) speedup vs IL0 %.3fx\n", nm[v], med[v], t[v].front(), t[v].back(), med[0] / med[v]);
    std::vector<float> o(2 * H); CK(cudaMemcpy(o.data(), obuf, 2 * H * 4, cudaMemcpyDeviceToHost));
    double cs = 0; int bad = 0; for (float x : o) { cs += x; bad += !(x == x); }
    printf("out checksum %.6e nan %d\n", cs, bad);
    return 0;
}
