// Card 28c: fused exp-on-load PV microbench (one KV group, one 64-query tile x 8 heads = 512 cols).
// current: split softmax (read S+mask, write P, stats) -> cuBLAS batched NN PV -> LSE reduce
// fused:   read-only split stats (m,l) -> pvexp GEMM computing exp(s+mask-m) while loading B -> same reduce
// Also reports pvexp alone vs cuBLAS batched alone (kernel speed fraction) and RMSE of both vs FP64.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
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
template <bool WRITE>
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
    for (int i = 0; i < PT; ++i) { int k = ks + threadIdx.x + i * FT; float p = __expf(v[i] - M); s += p; if (WRITE && k < ke) c[k] = p; }
    for (int o = 16; o; o /= 2) s += __shfl_xor_sync(~0u, s, o);
    if (!lane) ws[w] = s; __syncthreads();
    if (!threadIdx.x) { float t = 0; for (int j = 0; j < FT / 32; ++j) t += ws[j]; stats[size_t(col) * scap + split] = make_float2(M, t); }
}
__global__ void red(const float * part, const float2 * stats, float * out, int ns, int scap, int sub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; const int E = NC * D; if (i >= E) return;
    int col = i / D; const float2 * st = stats + size_t(col) * scap; float M = -FLT_MAX;
    for (int s = 0; s < ns; ++s) if (st[s].y > 0) M = fmaxf(M, st[s].x);
    float sum = 0, tot = 0;
    for (int s = 0; s < ns; ++s) if (st[s].y > 0) { float w = expf(st[s].x - M); sum += w * st[s].y; float pp = 0; for (int p = 0; p < sub; ++p) pp += part[size_t(s * sub + p) * E + i]; tot += w * pp; }
    out[i] = sum > 0 ? tot / sum : 0;
}
__global__ void ptrs(const float * V, const float * S, float * P, const float ** l, const float ** r, float ** o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    l[i] = V + size_t(i) * SK * D; r[i] = S + size_t(i) * SK; o[i] = P + size_t(i) * NC * D;
}
// out[split][col][d] = sum_{k in split} V[k][d] * exp(S[col][k] + mask[q][k] - m[col][split]); BM=128 (d), BN=128 (col), BK=8,
// 256 threads, 8x8 per thread, register-prefetch double buffering.
#define BM 128
#define BN 128
#define BK 8
__global__ void __launch_bounds__(256, 2) pvexp(const float * __restrict__ V, const float * __restrict__ S, const half * __restrict__ mask,
                                                const float2 * __restrict__ stats, float * __restrict__ out, int keys, int scap, int sub) {
    const int bm = blockIdx.x * BM, bn = blockIdx.y * BN, split = blockIdx.z / sub, part = blockIdx.z % sub, t = threadIdx.x;
    const int per = (SK / BK + sub - 1) / sub, s0 = part * per, s1 = min(SK / BK, s0 + per);
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN + 4];
    const int ak = t / 32, ad = (t % 32) * 4;               // A: V row k, 4 d
    const int bc = t / 2, bk = (t % 2) * 4;                  // B: col bc, 4 keys
    const int col = bn + bc, q = col % TQ;
    const float mcol = stats[size_t(col) * scap + split].x;
    const size_t k0 = size_t(split) * SK + size_t(s0) * BK;
    const float * Ap = V + (k0 + ak) * D + bm + ad;
    const float * Bp = S + size_t(col) * keys + k0 + bk;
    const half * Mp = mask + size_t(q) * keys + k0 + bk;
    float4 ra = *reinterpret_cast<const float4 *>(Ap);
    float4 rb = *reinterpret_cast<const float4 *>(Bp);
    uint2 rm = *reinterpret_cast<const uint2 *>(Mp);
    float acc[8][8] = {};
    const int tx = t % 16, ty = t / 16;
    auto stash = [&](int buf) {
        *reinterpret_cast<float4 *>(&As[buf][ak][ad]) = ra;
        const half2 m01 = *reinterpret_cast<const half2 *>(&rm.x), m23 = *reinterpret_cast<const half2 *>(&rm.y);
        const float2 f01 = __half22float2(m01), f23 = __half22float2(m23);
        Bs[buf][bk + 0][bc] = __expf(rb.x + f01.x - mcol);
        Bs[buf][bk + 1][bc] = __expf(rb.y + f01.y - mcol);
        Bs[buf][bk + 2][bc] = __expf(rb.z + f23.x - mcol);
        Bs[buf][bk + 3][bc] = __expf(rb.w + f23.y - mcol);
    };
    stash(0);
    __syncthreads();
    const int steps = s1 - s0;
    for (int s = 0; s < steps; ++s) {
        const int buf = s & 1;
        if (s + 1 < steps) {
            ra = *reinterpret_cast<const float4 *>(Ap + size_t(s + 1) * BK * D);
            rb = *reinterpret_cast<const float4 *>(Bp + (s + 1) * BK);
            rm = *reinterpret_cast<const uint2 *>(Mp + (s + 1) * BK);
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4 *>(&As[buf][k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&As[buf][k][64 + ty * 4]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&Bs[buf][k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&Bs[buf][k][64 + tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (s + 1 < steps) stash(buf ^ 1);
        __syncthreads();
    }
    float * o = out + size_t(blockIdx.z) * NC * D;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = bn + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
        *reinterpret_cast<float4 *>(o + size_t(c) * D + bm + ty * 4) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
        *reinterpret_cast<float4 *>(o + size_t(c) * D + bm + 64 + ty * 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
    }
}
int main(int argc, char ** argv) {
    int list[] = {92160, 61440, 30720};
    cublasHandle_t h; cublasCreate(&h);
    for (int keys : list) {
        const int ns = keys / SK, scap = ns;
        std::mt19937 rng(3); std::normal_distribution<float> nd(0, 1);
        std::vector<float> hV(size_t(keys) * D), hS(size_t(keys) * NC);
        for (auto & x : hV) x = nd(rng);
        for (auto & x : hS) x = 2.0f * nd(rng);  // score spread like scaled QK
        std::vector<half> hM(size_t(TQ) * keys);
        for (int q = 0; q < TQ; ++q) for (int k = 0; k < keys; ++k) hM[size_t(q) * keys + k] = __float2half(k >= keys - TQ + q + 1 ? -INFINITY : 0.f); // causal tail
        float *V, *S, *S2, *P1, *P2, *O1, *O2; float2 *st1, *st2; half * Mk; const float ** pl; const float ** pr; float ** po;
        cudaMalloc(&V, hV.size() * 4); cudaMalloc(&S, hS.size() * 4); cudaMalloc(&S2, hS.size() * 4); cudaMalloc(&Mk, hM.size() * 2);
        cudaMalloc(&P1, size_t(ns) * NC * D * 4); cudaMalloc(&P2, size_t(ns) * 8 * NC * D * 4); cudaMalloc(&O1, NC * D * 4); cudaMalloc(&O2, NC * D * 4);
        cudaMalloc(&st1, size_t(NC) * scap * 8); cudaMalloc(&st2, size_t(NC) * scap * 8);
        cudaMalloc(&pl, ns * 8); cudaMalloc(&pr, ns * 8); cudaMalloc(&po, ns * 8);
        cudaMemcpy(V, hV.data(), hV.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(S, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(Mk, hM.data(), hM.size() * 2, cudaMemcpyHostToDevice);
        ptrs<<<1, 256>>>(V, S2, P1, pl, pr, po, ns);
        const float one = 1, zero = 0;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        auto tim = [&](auto f, int n) { f(); cudaEventRecord(a); for (int i = 0; i < n; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms / n; };
        int SUB = 1;
        auto cur = [&] {  // S2 is the in-place P buffer; refresh from S first (copy not timed separately: see cur_copy)
            soft<true><<<dim3(GH, ns, TQ), FT>>>(S2, st1, Mk, keys, scap);
            cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, NC, SK, &one, pl, D, pr, keys, &zero, po, D, ns);
            red<<<NC * D / 256, 256>>>(P1, st1, O1, ns, scap, 1);
        };
        auto cpy = [&] { cudaMemcpyAsync(S2, S, hS.size() * 4, cudaMemcpyDeviceToDevice); };
        auto fus = [&] {
            soft<false><<<dim3(GH, ns, TQ), FT>>>(S, st2, Mk, keys, scap);
            pvexp<<<dim3(D / BM, NC / BN, ns * SUB), 256>>>(V, S, Mk, st2, P2, keys, scap, SUB);
            red<<<NC * D / 256, 256>>>(P2, st2, O2, ns, scap, SUB);
        };
        auto gemm_cub = [&] { cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, NC, SK, &one, pl, D, pr, keys, &zero, po, D, ns); };
        auto gemm_fus = [&] { pvexp<<<dim3(D / BM, NC / BN, ns * SUB), 256>>>(V, S, Mk, st2, P2, keys, scap, SUB); };
        for (int sb : {1, 2, 3, 4, 6}) { SUB = sb;
        for (int run = 0; run < 2; ++run) {
            float tc = tim([&] { cpy(); cur(); }, 10) - tim(cpy, 10);
            float tf = tim(fus, 10);
            float gc = tim(gemm_cub, 10), gf = tim(gemm_fus, 10);
            float ss = tim([&] { soft<true><<<dim3(GH, ns, TQ), FT>>>(S2, st1, Mk, keys, scap); }, 10);
            float sr = tim([&] { soft<false><<<dim3(GH, ns, TQ), FT>>>(S, st2, Mk, keys, scap); }, 10);
            const double fl = 2.0 * D * NC * double(ns) * SK;
            printf("keys=%d sub%d run%d current %.3f ms fused %.3f ms ratio %.3f | gemm cuBLAS %.3f (%.2f TF) pvexp %.3f (%.2f TF) speed %.1f%% | softmax_w %.3f stats_ro %.3f err=%s\n",
                   keys, SUB, run, tc, tf, tf / tc, gc, fl / gc / 1e9, gf, fl / gf / 1e9, 100 * gc / gf, ss, sr, cudaGetErrorString(cudaGetLastError()));
        }
        }
        SUB = 3;
        // FP64 reference on sampled cols
        cpy(); cur(); fus(); cudaDeviceSynchronize();
        std::vector<float> o1(NC * D), o2(NC * D);
        cudaMemcpy(o1.data(), O1, o1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(o2.data(), O2, o2.size() * 4, cudaMemcpyDeviceToHost);
        double e1 = 0, e2 = 0, r = 0; int n = 0; const int kk = ns * SK;
        for (int col = 0; col < NC; col += 37) {
            const int q = col % TQ; double mx = -1e300; std::vector<double> p(kk);
            for (int k = 0; k < kk; ++k) { double x = hS[size_t(col) * keys + k] + (double) __half2float(hM[size_t(q) * keys + k]); p[k] = x; if (x > mx) mx = x; }
            double sum = 0; for (int k = 0; k < kk; ++k) { p[k] = std::isinf(p[k]) ? 0 : std::exp(p[k] - mx); sum += p[k]; }
            for (int d = 0; d < D; ++d) { double acc = 0; for (int k = 0; k < kk; ++k) acc += p[k] * hV[size_t(k) * D + d]; acc /= sum;
                double x1 = o1[col * D + d] - acc, x2 = o2[col * D + d] - acc; e1 += x1 * x1; e2 += x2 * x2; r += acc * acc; ++n; }
        }
        printf("keys=%d rmse_vs_fp64 (fused sub3) current %.4g fused %.4g (rms ref %.4g, n=%d)\n", keys, std::sqrt(e1 / n), std::sqrt(e2 / n), std::sqrt(r / n), n);
        cudaFree(V); cudaFree(S); cudaFree(S2); cudaFree(Mk); cudaFree(P1); cudaFree(P2); cudaFree(O1); cudaFree(O2); cudaFree(st1); cudaFree(st2); cudaFree(pl); cudaFree(pr); cudaFree(po);
    }
    return 0;
}
