// Card 28b: does overlapping the memory-bound split softmax with the next tile's QK GEMM (2 streams,
// double-buffered scores) hide the score round-trip? Same kernels/shapes as fattn-maxwell-gemm.cuh for one
// KV group: per 64-query tile QK (TN cuBLAS) -> fused split softmax -> PV batched (NN) -> LSE reduce.
// Prints serial vs pipelined ms per tile and max |out_serial - out_pipe| (expected 0: same ops, same order).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <random>
#define D 256
#define NC 512
#define SK 4096
#define TQ 64
#define GH 8
#define FT 256
#define PT (SK / FT)
__global__ void __launch_bounds__(FT) soft(float * scores, float2 * stats, const half * mask, int keys, int scap) {
    const int head = blockIdx.x, split = blockIdx.y, q = blockIdx.z, col = head * TQ + q;
    const int ks = split * SK, ke = min(keys, ks + SK);
    float * c = scores + size_t(col) * keys; const half * m = mask + size_t(q) * keys;
    float v[PT]; float mx = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < PT; ++i) { int k = ks + threadIdx.x + i * FT; float x = -INFINITY; if (k < ke) x = c[k] + __half2float(m[k]); v[i] = x; mx = fmaxf(mx, x); }
    __shared__ float wm[FT / 32], ws[FT / 32]; int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    for (int o = 16; o; o /= 2) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    if (!lane) wm[w] = mx; __syncthreads(); float M = wm[0]; for (int j = 1; j < FT / 32; ++j) M = fmaxf(M, wm[j]);
    float s = 0;
#pragma unroll
    for (int i = 0; i < PT; ++i) { int k = ks + threadIdx.x + i * FT; float p = __expf(v[i] - M); s += p; if (k < ke) c[k] = p; }
    for (int o = 16; o; o /= 2) s += __shfl_xor_sync(~0u, s, o);
    if (!lane) ws[w] = s; __syncthreads();
    if (!threadIdx.x) { float t = 0; for (int j = 0; j < FT / 32; ++j) t += ws[j]; stats[size_t(col) * scap + split] = make_float2(M, t); }
}
__global__ void red(const float * part, const float2 * stats, float * out, int ns, int scap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; const int E = NC * D; if (i >= E) return;
    int col = i / D; const float2 * st = stats + size_t(col) * scap; float M = -FLT_MAX;
    for (int s = 0; s < ns; ++s) if (st[s].y > 0) M = fmaxf(M, st[s].x);
    float sum = 0, tot = 0;
    for (int s = 0; s < ns; ++s) if (st[s].y > 0) { float w = expf(st[s].x - M); sum += w * st[s].y; tot += w * part[size_t(s) * E + i]; }
    out[i] = sum > 0 ? tot / sum : 0;
}
__global__ void ptrs(const float * V, const float * S, float * P, const float ** l, const float ** r, float ** o, int n, int keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    l[i] = V + size_t(i) * SK * D; r[i] = S + size_t(i) * SK; o[i] = P + size_t(i) * NC * D;
}
int main() {
    int list[] = {30720, 61440, 92160}; const int T = 8; // 8 tiles = 512 queries per head group
    cublasHandle_t h; cublasCreate(&h);
    for (int keys : list) {
        const int ns = keys / SK, scap = ns;
        std::mt19937 rng(2); std::normal_distribution<float> nd(0, 1);
        std::vector<float> hK(size_t(keys) * D), hV(hK.size()), hQ(size_t(T) * NC * D);
        for (auto & x : hK) x = nd(rng); for (auto & x : hV) x = nd(rng); for (auto & x : hQ) x = nd(rng);
        std::vector<half> hM(size_t(TQ) * keys, __float2half(0.f));
        float *K, *V, *Q, *S[2], *P[2], *O[2][T]; float2 * st[2]; half * Mk; const float ** pl[2]; const float ** pr[2]; float ** po[2];
        cudaMalloc(&K, hK.size() * 4); cudaMalloc(&V, hV.size() * 4); cudaMalloc(&Q, hQ.size() * 4); cudaMalloc(&Mk, hM.size() * 2);
        cudaMemcpy(K, hK.data(), hK.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(V, hV.data(), hV.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(Q, hQ.data(), hQ.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(Mk, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice);
        for (int b = 0; b < 2; ++b) {
            cudaMalloc(&S[b], size_t(keys) * NC * 4); cudaMalloc(&P[b], size_t(ns) * NC * D * 4); cudaMalloc(&st[b], size_t(NC) * scap * 8);
            cudaMalloc(&pl[b], ns * 8); cudaMalloc(&pr[b], ns * 8); cudaMalloc(&po[b], ns * 8);
            ptrs<<<1, 256>>>(V, S[b], P[b], pl[b], pr[b], po[b], ns, keys);
            for (int t = 0; t < T; ++t) cudaMalloc(&O[b][t], size_t(NC) * D * 4);
        }
        cudaStream_t s0, s1; cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaEvent_t eq[T], ed[T], a, b; for (int t = 0; t < T; ++t) { cudaEventCreateWithFlags(&eq[t], cudaEventDisableTiming); cudaEventCreateWithFlags(&ed[t], cudaEventDisableTiming); }
        cudaEventCreate(&a); cudaEventCreate(&b);
        const float sc = 0.0625f, one = 1, zero = 0;
        auto qk = [&](int t, int bf, cudaStream_t s) { cublasSetStream(h, s); cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, NC, D, &sc, K, D, Q + size_t(t) * NC * D, D, &zero, S[bf], keys); };
        auto rest = [&](int t, int bf, float * out, cudaStream_t s) {
            soft<<<dim3(GH, ns, TQ), FT, 0, s>>>(S[bf], st[bf], Mk, keys, scap);
            cublasSetStream(h, s); cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, NC, SK, &one, pl[bf], D, pr[bf], keys, &zero, po[bf], D, ns);
            red<<<NC * D / 256, 256, 0, s>>>(P[bf], st[bf], out, ns, scap);
        };
        auto serial = [&] { for (int t = 0; t < T; ++t) { qk(t, 0, s0); rest(t, 0, O[0][t], s0); } };
        auto pipe = [&] {
            for (int t = 0; t < T; ++t) {
                const int bf = t & 1;
                if (t >= 2) cudaStreamWaitEvent(s0, ed[t - 2], 0);
                qk(t, bf, s0); cudaEventRecord(eq[t], s0);
                cudaStreamWaitEvent(s1, eq[t], 0); rest(t, bf, O[1][t], s1); cudaEventRecord(ed[t], s1);
            }
            cudaEventRecord(eq[0], s1); cudaStreamWaitEvent(s0, eq[0], 0);
        };
        float tm[2] = {0, 0};
        for (int rep = 0; rep < 5; ++rep) for (int v = 0; v < 2; ++v) {
            cudaDeviceSynchronize(); cudaEventRecord(a, s0); if (v) pipe(); else serial(); cudaEventRecord(b, s0); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); if (rep) tm[v] += ms / 4 / T;
        }
        double md = 0; std::vector<float> o0(size_t(NC) * D), o1(o0.size());
        for (int t = 0; t < T; ++t) { cudaMemcpy(o0.data(), O[0][t], o0.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(o1.data(), O[1][t], o1.size() * 4, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < o0.size(); ++i) md = std::max(md, (double) std::fabs(o0[i] - o1[i])); }
        printf("keys=%d serial %.3f ms/tile pipelined %.3f ms/tile gain %.2f%% max|diff| %.3g err=%s\n", keys, tm[0], tm[1], 100 * (1 - tm[1] / tm[0]), md, cudaGetErrorString(cudaGetLastError()));
        cudaFree(K); cudaFree(V); cudaFree(Q); cudaFree(Mk);
        for (int bb = 0; bb < 2; ++bb) { cudaFree(S[bb]); cudaFree(P[bb]); cudaFree(st[bb]); cudaFree(pl[bb]); cudaFree(pr[bb]); cudaFree(po[bb]); for (int t = 0; t < T; ++t) cudaFree(O[bb][t]); }
    }
    return 0;
}
