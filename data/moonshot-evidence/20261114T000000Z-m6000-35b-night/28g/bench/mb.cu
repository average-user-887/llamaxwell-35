// Card 28g microbench: occamy moe_v3 (decode, one token per launch as in prod) on synthetic Q4_0 weights.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include KINC
}  // namespace occamy
using namespace occamy;
__device__ unsigned hsh(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }
__global__ void fillq(uint8_t * p, size_t nblk, unsigned seed) {  // Q4_0 blocks: half d ~ U(0.005,0.03), random nibbles
    for (size_t b = blockIdx.x * (size_t)blockDim.x + threadIdx.x; b < nblk; b += (size_t)gridDim.x * blockDim.x) {
        unsigned h = hsh((unsigned)b * 2654435761u ^ seed);
        __half d = __float2half(0.005f + 0.025f * (h & 0xffff) / 65535.f);
        uint8_t * q = p + b * 18; memcpy(q, &d, 2);
        for (int i = 0; i < 16; ++i) q[2 + i] = (uint8_t)hsh(h + i);
    }
}
__global__ void fillf(float * p, size_t n, unsigned seed, float sc) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = sc * ((hsh((unsigned)i ^ seed) & 0xffffff) / 16777215.f * 2.f - 1.f);
}
int main(int argc, char ** argv) {
    const int NH = 64;
    size_t nGU = (size_t)NE * FF * ROW_GU, nD = (size_t)NE * H * ROW_D;
    uint8_t *ge, *ue, *de, *gs, *us, *ds; float *gi, *gis, *h, *nw, *out, *out2; Scratch * sc;
    cudaMalloc(&ge, nGU); cudaMalloc(&ue, nGU); cudaMalloc(&de, nD);
    cudaMalloc(&gs, (size_t)FFS * ROW_GU); cudaMalloc(&us, (size_t)FFS * ROW_GU); cudaMalloc(&ds, (size_t)H * (FFS / 32 * QB));
    cudaMalloc(&gi, (size_t)NE * H * 4); cudaMalloc(&gis, H * 4); cudaMalloc(&h, (size_t)NH * H * 4); cudaMalloc(&nw, H * 4);
    cudaMalloc(&out, (size_t)NH * H * 4); cudaMalloc(&out2, (size_t)NH * H * 4); cudaMalloc(&sc, sizeof(Scratch)); cudaMemset(sc, 0, sizeof(Scratch));
    fillq<<<1024, 256>>>(ge, nGU / 18, 1); fillq<<<1024, 256>>>(ue, nGU / 18, 2); fillq<<<1024, 256>>>(de, nD / 18, 3);
    fillq<<<256, 256>>>(gs, (size_t)FFS * ROW_GU / 18, 4); fillq<<<256, 256>>>(us, (size_t)FFS * ROW_GU / 18, 5); fillq<<<256, 256>>>(ds, (size_t)H * 288 / 18, 6);
    fillf<<<1024, 256>>>(gi, (size_t)NE * H, 7, 0.05f); fillf<<<8, 256>>>(gis, H, 8, 0.05f); fillf<<<512, 256>>>(h, (size_t)NH * H, 9, 1.f);
    fillf<<<8, 256>>>(nw, H, 10, 1.f);
    cudaDeviceSynchronize();
    Params P; P.L.gate_inp = gi; P.L.gate_inp_sh = gis; P.L.gate_exps = ge; P.L.up_exps = ue; P.L.down_exps = de;
    P.L.gate_sh = gs; P.L.up_sh = us; P.L.down_sh = ds; P.norm = nw; P.s = sc; P.eps = 1e-6f; P.tsrow = -1;
    int dev; cudaGetDevice(&dev); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
    int per = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, moe_v3<true>, NT3, SMEM3_FLOATS * sizeof(float));
    const int gmax = per * pr.multiProcessorCount;
  for (int grid : {gmax, 40, 36, 32, 24}) {
    printf("grid %d (%d/SM), smem %zu\n", grid, per, SMEM3_FLOATS * sizeof(float));
    auto run = [&](int i, float * o) { P.h = h + (size_t)(i % NH) * H; P.out = o + (size_t)(i % NH) * H; moe_v3<true><<<grid, NT3, SMEM3_FLOATS * sizeof(float)>>>(P); };
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int rep = 0; rep < 2; ++rep) {
        for (int i = 0; i < 16; ++i) run(i, out);
        cudaEventRecord(a); const int N = 256; for (int i = 0; i < N; ++i) run(i * 7, out); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("grid %d rep%d moe_v3 %.2f us/launch (%s)\n", grid, rep, 1000 * ms / N, cudaGetErrorString(cudaGetLastError()));
    }
    // phase stamps on one launch
    P.tsrow = 0; run(5, out2); cudaDeviceSynchronize(); P.tsrow = -1;
    unsigned long long ts[64][8]; cudaMemcpyFromSymbol(ts, g_ts, sizeof(ts));
    unsigned long long t0 = ~0ull, mx[6] = {0}, mn[6]; for (int k = 0; k < 6; ++k) mn[k] = ~0ull;
    for (int bk = 0; bk < grid; ++bk) for (int k = 0; k < 6; ++k) { mx[k] = std::max(mx[k], ts[bk][k]); mn[k] = std::min(mn[k], ts[bk][k]); }
    t0 = mn[0];
    printf("phase stamps (us from first block start, min/max over blocks): start %.1f/%.1f A-end %.1f/%.1f bar1-out %.1f/%.1f B-end %.1f/%.1f bar2-out %.1f/%.1f C-end %.1f/%.1f\n",
           (mn[0]-t0)/1e3,(mx[0]-t0)/1e3,(mn[1]-t0)/1e3,(mx[1]-t0)/1e3,(mn[2]-t0)/1e3,(mx[2]-t0)/1e3,(mn[3]-t0)/1e3,(mx[3]-t0)/1e3,(mn[4]-t0)/1e3,(mx[4]-t0)/1e3,(mn[5]-t0)/1e3,(mx[5]-t0)/1e3);
  }
    return 0;
}
