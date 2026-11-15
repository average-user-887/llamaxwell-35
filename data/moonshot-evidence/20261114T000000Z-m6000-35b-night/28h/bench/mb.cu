// Card 28h: gdn_fused2 decode microbench (real shapes, synthetic data, in-place state rows as in prod N2GDN).
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#ifndef KINC
#define KINC "kern_ts.inc"
#endif
#include KINC
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
int main(int argc, char **argv) {
    int dev = 0; cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    int per_sm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_fused2, NT, SMEM3_FLOATS * sizeof(float)));
    const int gmax = per_sm * pr.multiProcessorCount;
    printf("per_sm %d grid %d smem %zu Scratch %zu\n", per_sm, gmax, SMEM3_FLOATS * sizeof(float), sizeof(Scratch));
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
    CK(cudaFuncSetAttribute(gdn_fused2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM3_FLOATS * sizeof(float)));
    const size_t bytes = (size_t)NROWS_A * ROW_H + (size_t)H * ROW_O + 2 * 2 * (size_t)SSM_N * 4 + 2 * 2 * (size_t)CONV_N * 4;
    int grids[3] = { gmax, 66, 64 };
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int gi = 0; gi < 3; ++gi) {
        const int g = grids[gi]; if (g > gmax || g * NW * N2R < H) continue;
        int on = 0; CK(cudaMemcpyToSymbol(g_tson, &on, 4));
        for (int i = 0; i < 16; ++i) gdn_fused2<<<g, NT, SMEM3_FLOATS * sizeof(float)>>>(PP);
        CK(cudaDeviceSynchronize());
        for (int rep = 0; rep < 2; ++rep) {
            CK(cudaEventRecord(e0)); for (int i = 0; i < 256; ++i) gdn_fused2<<<g, NT, SMEM3_FLOATS * sizeof(float)>>>(PP);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            const float us = ms * 1000.f / 256; printf("grid %d rep%d gdn_fused2 %.2f us/launch  %.1f MB  %.0f GB/s\n", g, rep, us, bytes / 1e6, bytes / (us * 1e3));
        }
        on = 1; CK(cudaMemcpyToSymbol(g_tson, &on, 4));
        unsigned long long acc[8][2] = {}; const int NR = 8;
        for (int r = 0; r < NR; ++r) {
            gdn_fused2<<<g, NT, SMEM3_FLOATS * sizeof(float)>>>(PP); CK(cudaDeviceSynchronize());
            unsigned long long ts[96][8]; CK(cudaMemcpyFromSymbol(ts, g_ts, sizeof(ts)));
            unsigned long long t0 = ~0ull; for (int b = 0; b < g; ++b) t0 = ts[b][0] < t0 ? ts[b][0] : t0;
            for (int n = 0; n < 8; ++n) { unsigned long long mn = ~0ull, mx = 0; for (int b = 0; b < g; ++b) { unsigned long long v = ts[b][n] - t0; mn = v < mn ? v : mn; mx = v > mx ? v : mx; } acc[n][0] += mn; acc[n][1] += mx; }
        }
        const char *nm[8] = { "start", "A-end", "bar1-out", "B0-end", "B1-end", "bar2-out", "Cpass1-end", "C-end" };
        printf("grid %d phase stamps (us, min/max over blocks, mean of %d):", g, NR);
        for (int n = 0; n < 8; ++n) printf(" %s %.1f/%.1f", nm[n], acc[n][0] / 1e3 / NR, acc[n][1] / 1e3 / NR);
        printf("\n");
    }
    std::vector<float> o(2 * H); CK(cudaMemcpy(o.data(), obuf, 2 * H * 4, cudaMemcpyDeviceToHost));
    double cs = 0; int bad = 0; for (float v : o) { cs += v; bad += !(v == v); }
    printf("out checksum %.6e nan %d\n", cs, bad);
    return 0;
}
