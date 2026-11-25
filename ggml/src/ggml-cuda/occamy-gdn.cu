// native-occamy Track G (13 Oct 2026): fused qwen35moe GDN layer DECODE segment (batch 1, one sequence), sm_52.
// Kernel = native-occamy/decode/gdn_fused.cu gdn_fused<PHASE, V=1> (d2 staged uint4 row loads), adapted to the
// ggml graph: the recurrent states are read from the get_rows source row (s_copy[0]) and written to the cpy
// destinations (kv_head rows); the q/k conv-state write moves to phase C because q/k channels are read by 16
// phase-B items (conv in/out alias in-graph). Activations stay fp32 (no q8_1 requantisation).
//   A: xn = rmsnorm(h)*attn_norm (per block); 12352 GEMV rows (qkv 8192, gate 4096, alpha 32, beta 32), K=2048
//   B: per (v head, 16-column group): conv1d(4)+silu, l2norm q/k, gate/beta, delta rule on 2 columns per warp
//   C: per-head rmsnorm * ssm_norm * silu(z) -> ssm_out GEMV (K=4096) + residual; q/k conv state write
#include "occamy-gdn.cuh"
#include "sm52-xmad.cuh"
#include "gated_delta_net.cuh"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace occamy_gdn {

constexpr int H = 2048, NQKV = 8192, DI = 4096, NVH = 32, NKH = 16, SD = 128;
constexpr int QB = 18;
constexpr int ROW_H = H / 32 * QB;     // 1152 B (K=2048)
constexpr int ROW_O = DI / 32 * QB;    // 2304 B (K=4096)
constexpr int NT = 256, NW = NT / 32;
constexpr int CH = 68;
constexpr int NCG = 8, CGC = SD / NCG; // column groups per head (16 cols each, 2 per warp)
constexpr int NROWS_A = NQKV + DI + 2 * NVH;
constexpr int CONV_N = 3 * NQKV;       // conv state floats per row
constexpr int SSM_N  = NVH * SD * SD;  // ssm state floats per row
constexpr unsigned SPIN_LIMIT = 1u << 26;

struct Layer {
    const float * attn_norm, * conv_w, * ssm_a, * dt_bias, * ssm_norm;
    const uint8_t * qkv, * gate, * alpha, * beta, * out;
};
struct Scratch {
    float qkv[NQKV], z[DI], ab[2 * NVH], attn[DI], ssq[NVH * NCG];
    unsigned bar_count, bar_gen;
    // C11 (PCS): shared phase-C preparation, written once per head in phase B by the last contributing block
    alignas(16) float xg[DI]; alignas(16) float sg[DI / 32];   // gated/normalised phase-C input and its 32-element block sums
    alignas(16) uint4 q8c_pk[16 * 32];   // Q8C: packed Q8_1 phase-C input (2 halves * 8 * 32 uint4s = 8192 B)
    alignas(16) float2 q8c_ds[DI / 32];  // Q8C: (d, sum) per 32-element block (128 float2s = 1024 B)
    unsigned hcnt[NVH];          // per-head phase-B contribution counter, +NCG per launch (multiple of NCG between launches)
};
struct Params {
    Layer L;
    const float * h;          // residual stream in [2048]
    float * out;              // [2048] = h + ssm_out(...)
    const int32_t * s_copy;   // source row index for both states
    const float * conv_base; size_t conv_stride;   // [row][8192][3] (time-fastest, oldest first)
    float * conv_out;                              // [8192][3]
    const float * ssm_base;  size_t ssm_stride;    // [row][32][128 col][128 i] (transposed, as ggml-cuda)
    float * S_out;                                 // [32][128][128]
    Scratch * s;
    float eps_attn, eps_l2, scale_l2, eps_gated;
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ void grid_barrier(Scratch * s) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned * genp = &s->bar_gen;
        unsigned gen = *genp;
        __threadfence();
        if (atomicAdd(&s->bar_count, 1) == gridDim.x - 1) {
            s->bar_count = 0;
            __threadfence();
            atomicAdd(&s->bar_gen, 1);
        } else {
            unsigned k = 0;
            while (*genp == gen) { if (++k > SPIN_LIMIT) { __trap(); } }
        }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ float nib(unsigned v, int k) {
    return __int_as_float(__byte_perm(v, 0x3F000000u, 0x7044u | (k << 8)));
}
// dot of 2 consecutive Q4_0 blocks staged as 9 smem words (lane stride 9) with a 64-float smem chunk and its block sums
__device__ __forceinline__ float dot2v(unsigned w0, unsigned w1, unsigned w2, unsigned w3, unsigned w4, unsigned w5, unsigned w6,
                                       unsigned w7, unsigned w8, const float * xc, float2 sx) {
    float d0 = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
    float d1 = __half2float(__ushort_as_half((unsigned short)(w4 >> 16)));
    unsigned q0[4] = { __byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432), __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432) };
    unsigned q1[4] = { w5, w6, w7, w8 };
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned lo = (q0[i] & 0x0F0F0F0Fu) | 0x80808080u, hi = ((q0[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        float4 xl = *(const float4 *)(xc + 4 * i), xh = *(const float4 *)(xc + 16 + 4 * i);
        a0 = fmaf(nib(lo, 0), xl.x, a0); a0 = fmaf(nib(lo, 1), xl.y, a0); a0 = fmaf(nib(lo, 2), xl.z, a0); a0 = fmaf(nib(lo, 3), xl.w, a0);
        a0 = fmaf(nib(hi, 0), xh.x, a0); a0 = fmaf(nib(hi, 1), xh.y, a0); a0 = fmaf(nib(hi, 2), xh.z, a0); a0 = fmaf(nib(hi, 3), xh.w, a0);
        lo = (q1[i] & 0x0F0F0F0Fu) | 0x80808080u; hi = ((q1[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        xl = *(const float4 *)(xc + 32 + 4 * i); xh = *(const float4 *)(xc + 48 + 4 * i);
        a1 = fmaf(nib(lo, 0), xl.x, a1); a1 = fmaf(nib(lo, 1), xl.y, a1); a1 = fmaf(nib(lo, 2), xl.z, a1); a1 = fmaf(nib(lo, 3), xl.w, a1);
        a1 = fmaf(nib(hi, 0), xh.x, a1); a1 = fmaf(nib(hi, 1), xh.y, a1); a1 = fmaf(nib(hi, 2), xh.z, a1); a1 = fmaf(nib(hi, 3), xh.w, a1);
    }
    return d0 * (128.f * a0 - 136.f * sx.x) + d1 * (128.f * a1 - 136.f * sx.y);
}
__device__ __forceinline__ float dot2w(const unsigned * w, const float * xc, float2 sx) {
    return dot2v(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], xc, sx);
}
// GDN3 card B (REPACK): ssm_out row packed as a lossless 32-bit word permutation (576 words/row, same bytes).
// Lane l needs words 9l..9l+8 (chunk l) and 288+9l..+8 (chunk l+32) = local word j (0..17).
// Packed word q: q < 512 -> [g = q/128][lane][j%4] with j = 4g + q%4; q >= 512 -> [lane][2] with j = 16 + q%2.
__host__ __device__ __forceinline__ int rp_src(int q) {
    int lane, j;
    if (q < 512) { lane = (q & 127) >> 2; j = (q >> 7) * 4 + (q & 3); } else { lane = (q - 512) >> 1; j = 16 + (q & 1); }
    return j < 9 ? lane * 9 + j : 288 + lane * 9 + (j - 9);
}
__device__ __forceinline__ void ldrp(const uint8_t * base, int r, int lane, unsigned (&w)[18]) {
    const uint4 * p = (const uint4 *)(base + (size_t)r * (DI / 32 * QB));
#pragma unroll
    for (int g = 0; g < 4; ++g) { const uint4 v = __ldg(p + 32 * g + lane); w[4 * g] = v.x; w[4 * g + 1] = v.y; w[4 * g + 2] = v.z; w[4 * g + 3] = v.w; }
    const uint2 t = __ldg((const uint2 *)(p + 128) + lane); w[16] = t.x; w[17] = t.y;
}
__device__ __forceinline__ const uint8_t * rowA(const Layer & L, int t) {
    return t < NQKV ? L.qkv + (size_t)t * ROW_H : t < NQKV + DI ? L.gate + (size_t)(t - NQKV) * ROW_H
         : t < NQKV + DI + NVH ? L.alpha + (size_t)(t - NQKV - DI) * ROW_H : L.beta + (size_t)(t - NQKV - DI - NVH) * ROW_H;
}
__device__ __forceinline__ void block_sums(const float * xs, float2 * sums, int n) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < n / 32; b += NW) {
        float s = warp_sum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
        if (lane == 0) ((float *)sums)[b] = s;
    }
    __syncthreads();
}

constexpr int SMEM_FLOATS = DI / 64 * CH + DI / 32 + 64;
constexpr int STW = ROW_O / 4;   // 576 stage words per warp
constexpr int SMEM2_FLOATS = SMEM_FLOATS + NW * STW;
constexpr int STW_RP = ROW_H / 4;   // 288: REPACK phase C reads straight to registers, only phase A stages
constexpr int SMEM3_FLOATS = SMEM_FLOATS + NW * STW_RP;

// RQK (C10, opt-in GGML_CUDA_OCCAMY_GDN_QKREUSE=1): phase-B items are handed out as contiguous per-block chunks in
// (k head, v head of that k head, column group) order and the block keeps the l2-normalised q/k (smem qn/kn + the two
// scale registers) and the gate/beta scalars while the k head / v head is unchanged. Same per-item arithmetic and order,
// same item->output indices, no new sync point; q/k conv state is still only read in phase B (written in phase C).
// PCS (C11, opt-in GGML_CUDA_OCCAMY_GDN_PCSHARE=1): the phase-C input xs = attn*fac*ssm_norm*silu(z) and its block sums
// are built once per head instead of once per block. Each block counts its phase-B items per head (smem hc); after its
// item loop it publishes (all threads fence once, one atomicAdd per touched head). The block whose add completes the
// head's NCG items builds that head's 128 xs values and 4 sums (same expressions/order as phase C) into Scratch xg/sg.
// The existing phase-B->C grid barrier (or the kernel boundary in 3-launch mode) orders them before phase C, which then
// only copies xg/sg into smem. No new grid barrier, no new launch.
// PF (GDN1, opt-in GGML_CUDA_OCCAMY_GDN_PREF=1, fused single launch only): load-issue hoisting, no arithmetic change.
//   A: each warp's first K=2048 row is requested at kernel entry, before the rmsnorm prologue (weights are read-only).
//   A->B: the block's first phase-B state tile (2 float4/thread) and each warp's first ssm_out row (5 uint4) are
//         requested before grid barrier 1 (state is not written in phase A; ssm_out weights are read-only).
//   B: the next item's state tile is requested right after the current item's delta-rule loop (tiles are disjoint, so
//      the S_in/S_out alias of the in-place row is safe). After the first item the first ssm_out row is staged into the
//      warp's (phase-B-idle) stg buffer and the second row requested, so phase C starts with two rows in flight/staged.
//   Every dot product, reduction and store is the same expression in the same order -> bit-identical results.
template <int PHASE, int RQK = 0, int PCS = 0, int PF = 0, int RP = 0>
__global__ void __launch_bounds__(NT, RP ? 3 : 2) gdn_fused(Params P) {
    static_assert(!(PF && RP), "REPACK excludes PREF (phase-C staging is gone)");
    extern __shared__ float smem[];
    __shared__ float qn[SD], kn[SD], vv[CGC], red[NW], fac[NVH];
    __shared__ unsigned hc[PCS ? NVH : 1], fin[PCS ? NVH + 1 : 1];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + DI / 64 * CH);
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + (threadIdx.x >> 5) * (RP ? STW_RP : STW); uint4 * stg4 = (uint4 *)stg;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    Scratch * s = P.s;
    const Layer & L = P.L;
    const int row = __ldg(P.s_copy);
    const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
    const float * S_in    = P.ssm_base  + (size_t)row * P.ssm_stride;
    constexpr bool PFX = PF && PHASE == 0;
    uint4 cpf[5];                 // PF: ssm_out row prefetch (first row before barrier 1, second after the first B item)
    float4 stn[2];                // PF: next phase-B state tile
    bool staged = false;          // PF: first ssm_out row is in stg, cpf holds the second

    if (PHASE == 0 || PHASE == 1) {
        // ---- phase A: xn = rmsnorm(h)*w (per block), then all K=2048 GEMV rows ----
        uint4 c0, c1, c2;
        int t = gw;
        if (PFX && t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        float ss = 0.f;
        for (int i = threadIdx.x; i < H; i += NT) { float v = P.h[i]; xs[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
        ss = warp_sum(ss);
        if (lane == 0) red[warp] = ss;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < NW; ++i) tot += red[i];
        const float sc = rsqrtf(tot / H + P.eps_attn);
        for (int i = threadIdx.x; i < H; i += NT) xs[(i >> 6) * CH + (i & 63)] *= sc * __ldg(L.attn_norm + i);
        __syncthreads();
        block_sums(xs, sums, H);
        const float * xc = xs + lane * CH; const float2 sx = sums[lane];
        if (!PFX && t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        for (; t < NROWS_A; t += W) {
            stg4[lane] = c0; stg4[lane + 32] = c1; if (lane < 8) stg4[lane + 64] = c2;
            __syncwarp();
            const int tn = t + W;
            if (tn < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
            float a = warp_sum(dot2w(stg + lane * 9, xc, sx));
            if (lane == 0) { float * dst = t < NQKV ? &s->qkv[t] : t < NQKV + DI ? &s->z[t - NQKV] : &s->ab[t - NQKV - DI]; *dst = a; }
            __syncwarp();
        }
    }
    if (PFX) {
        // first ssm_out row of this warp and the first phase-B state tile of this block, before barrier 1
        if (gw < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)gw * ROW_O);
#pragma unroll
            for (int i = 0; i < 4; ++i) cpf[i] = __ldg(p + lane + 32 * i);
            if (lane < 16) cpf[4] = __ldg(p + lane + 128); }
        const int it0 = RQK ? (int)(((long long)blockIdx.x * (NVH * NCG)) / gridDim.x) : (int)blockIdx.x;
        const int it1 = RQK ? (int)(((long long)(blockIdx.x + 1) * (NVH * NCG)) / gridDim.x) : NVH * NCG;
        if (it0 < it1) {
            const int hv = RQK ? it0 / (2 * NCG) + NKH * ((it0 / NCG) & 1) : it0 / NCG;
            const int col0 = (it0 % NCG) * CGC + warp * 2;
            const float4 * si = (const float4 *)(S_in + ((size_t)hv * SD + col0) * SD) + lane;
            stn[0] = si[0]; stn[1] = si[SD / 4];
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 2) {
        // ---- phase B: conv + l2norm + gates + delta rule, item = (v head, column group) ----
        if (PCS && threadIdx.x < NVH) { hc[threadIdx.x] = 0; if (threadIdx.x == 0) fin[NVH] = 0; }   // first use after a sync
        const int it_lo = RQK ? (int)(((long long)blockIdx.x * (NVH * NCG)) / gridDim.x) : (int)blockIdx.x;
        const int it_hi = RQK ? (int)(((long long)(blockIdx.x + 1) * (NVH * NCG)) / gridDim.x) : NVH * NCG;
        const int it_st = RQK ? 1 : (int)gridDim.x;
        int kh_c = -1, hv_c = -1;                                   // block-uniform cache tags (RQK only)
        float qs = 0.f, ks = 0.f, gexp = 0.f, beta = 0.f;
        for (int it = it_lo; it < it_hi; it += it_st) {
            const int hv = RQK ? it / (2 * NCG) + NKH * ((it / NCG) & 1) : it / NCG;
            const int cg = it % NCG, kh = hv % NKH;   // ggml-cuda: q/k head = v head mod n_k_heads
            const bool need_qk = !RQK || kh != kh_c;   // block-uniform
            {
                const int t = threadIdx.x;
                if (need_qk) {
                const int c = t < SD ? kh * SD + t : H + kh * SD + (t - SD);
                const float * ci = conv_in + c * 3; const float * w = L.conv_w + c * 4;
                float xnew = __ldcg(&s->qkv[c]);
                float y = ci[0] * w[0] + ci[1] * w[1] + ci[2] * w[2] + xnew * w[3];
                y = silu(y);
                float sq = warp_sum(y * y);
                if (lane == 0) red[warp] = sq;
                if (t < SD) qn[t] = y; else kn[t - SD] = y;
                }
                if (t < CGC) {
                    const int cv = 2 * H + hv * SD + cg * CGC + t;   // unique owner -> conv state written here
                    const float * cvi = conv_in + cv * 3; const float * wv = L.conv_w + cv * 4;
                    float xv = __ldcg(&s->qkv[cv]);
                    float c1 = cvi[1], c2 = cvi[2];
                    float yv = cvi[0] * wv[0] + c1 * wv[1] + c2 * wv[2] + xv * wv[3];
                    vv[t] = silu(yv);
                    float * co = P.conv_out + cv * 3; co[0] = c1; co[1] = c2; co[2] = xv;
                }
            }
            __syncthreads();
            // l2 norm as ggml: rms_norm(x, eps/n) * 1/sqrt(n)
            if (need_qk) {
            qs = rsqrtf((red[0] + red[1] + red[2] + red[3]) / SD + P.eps_l2) * P.scale_l2;
            ks = rsqrtf((red[4] + red[5] + red[6] + red[7]) / SD + P.eps_l2) * P.scale_l2;
            kh_c = kh;
            }
            if (!RQK || hv != hv_c) {
            const float al = __ldcg(&s->ab[hv]) + __ldg(L.dt_bias + hv);
            const float sp = al > 20.f ? al : logf(1.f + expf(al));
            gexp = expf(sp * __ldg(L.ssm_a + hv));
            beta = 1.f / (1.f + expf(-__ldcg(&s->ab[NVH + hv])));
            hv_c = hv;
            }
            const float4 q4 = *(const float4 *)(qn + 4 * lane), k4r = *(const float4 *)(kn + 4 * lane);
            const float4 qv = make_float4(q4.x * qs, q4.y * qs, q4.z * qs, q4.w * qs);
            const float4 kv4 = make_float4(k4r.x * ks, k4r.y * ks, k4r.z * ks, k4r.w * ks);
            const int col0 = cg * CGC + warp * 2;
            const float4 * si = (const float4 *)(S_in + ((size_t)hv * SD + col0) * SD) + lane;
            float4 * so = (float4 *)(P.S_out + ((size_t)hv * SD + col0) * SD) + lane;
            float4 st[2];
            if (PFX) { st[0] = stn[0]; st[1] = stn[1]; } else { st[0] = si[0]; st[1] = si[SD / 4]; }
            float ssq = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float4 a = st[j];
                float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                float delta = (vv[warp * 2 + j] - gexp * kvd) * beta;
                a.x = gexp * a.x + kv4.x * delta; a.y = gexp * a.y + kv4.y * delta;
                a.z = gexp * a.z + kv4.z * delta; a.w = gexp * a.w + kv4.w * delta;
                float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                so[j * SD / 4] = a;
                if (lane == 0) s->attn[hv * SD + col0 + j] = att;
                ssq += att * att;
            }
            if (PFX && it + it_st < it_hi) {   // next item's (disjoint) state tile
                const int itn = it + it_st;
                const int hvn = RQK ? itn / (2 * NCG) + NKH * ((itn / NCG) & 1) : itn / NCG;
                const int coln = (itn % NCG) * CGC + warp * 2;
                const float4 * sin_ = (const float4 *)(S_in + ((size_t)hvn * SD + coln) * SD) + lane;
                stn[0] = sin_[0]; stn[1] = sin_[SD / 4];
            }
            __syncthreads();   // everyone done with red/qn/kn/vv
            if (lane == 0) red[warp] = ssq;
            __syncthreads();
            if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < NW; ++i) t += red[i]; s->ssq[hv * NCG + cg] = t; if (PCS) hc[hv]++; }
            __syncthreads();
            if (PFX && !staged) {   // stage the first ssm_out row (stg is idle in phase B), request the second
                staged = true;
                if (gw < H) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) stg4[lane + 32 * i] = cpf[i];
                    if (lane < 16) stg4[lane + 128] = cpf[4];
                    const int rn = gw + W;
                    if (rn < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)rn * ROW_O);
#pragma unroll
                        for (int i = 0; i < 4; ++i) cpf[i] = __ldg(p + lane + 32 * i);
                        if (lane < 16) cpf[4] = __ldg(p + lane + 128); }
                }
            }
        }
        if (PFX && !staged && gw < H) {   // block without phase-B items
            staged = true;
#pragma unroll
            for (int i = 0; i < 4; ++i) stg4[lane + 32 * i] = cpf[i];
            if (lane < 16) stg4[lane + 128] = cpf[4];
            const int rn = gw + W;
            if (rn < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)rn * ROW_O);
#pragma unroll
                for (int i = 0; i < 4; ++i) cpf[i] = __ldg(p + lane + 32 * i);
                if (lane < 16) cpf[4] = __ldg(p + lane + 128); }
        }
        if (PCS) {
            // publish: every attn/ssq writer (lane 0 of each warp, incl. thread 0) fences once, then one add per touched head
            __threadfence();
            __syncthreads();
            if (threadIdx.x < NVH && hc[threadIdx.x] != 0) {
                const unsigned c = hc[threadIdx.x];
                const unsigned old = atomicAdd(&s->hcnt[threadIdx.x], c);
                if ((old % NCG) + c == NCG) { fin[atomicAdd(&fin[NVH], 1u)] = threadIdx.x; }   // this block completed the head
            }
            __syncthreads();
            const unsigned nf = fin[NVH];
            if (nf) {
                __threadfence();   // acquire side: other blocks' attn/ssq for the finished heads
                for (unsigned f = threadIdx.x >> 7; f < nf; f += NT / SD) {   // two heads at a time (threads 0-127, 128-255)
                    const int hv = fin[f], t = threadIdx.x & (SD - 1), i = hv * SD + t;
                    float tt = 0.f;
#pragma unroll
                    for (int k = 0; k < NCG; ++k) tt += __ldcg(&s->ssq[hv * NCG + k]);
                    const float fh = rsqrtf(tt / SD + P.eps_gated);
                    const float v = __ldcg(&s->attn[i]) * fh * __ldg(L.ssm_norm + t) * silu(__ldcg(&s->z[i]));
                    s->xg[i] = v;
                    const float bs = warp_sum(v);            // == block_sums(): sum b = elements 32b..32b+31, same lanes
                    if (lane == 0) s->sg[i >> 5] = bs;
                }
            }
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 3) {
        // q/k conv state: every phase-B reader is done, one writer per channel
        for (int c = blockIdx.x * NT + threadIdx.x; c < 2 * H; c += gridDim.x * NT) {
            const float * ci = conv_in + c * 3; float * co = P.conv_out + c * 3;
            const float a = ci[1], b = ci[2];
            co[0] = a; co[1] = b; co[2] = __ldcg(&s->qkv[c]);
        }
        // ---- phase C: o = rmsnorm_head(attn)*ssm_norm*silu(z) -> ssm_out GEMV + residual ----
        if (PCS) {
            for (int i = 4 * threadIdx.x; i < DI; i += 4 * NT)
                *(float4 *)(xs + (i >> 6) * CH + (i & 63)) = __ldcg((const float4 *)(s->xg + i));
            if (threadIdx.x < DI / 128) ((float4 *)sums)[threadIdx.x] = __ldcg((const float4 *)(s->sg) + threadIdx.x);
            __syncthreads();
        } else {
        if (threadIdx.x < NVH) {
            float t = 0.f;
#pragma unroll
            for (int i = 0; i < NCG; ++i) t += __ldcg(&s->ssq[threadIdx.x * NCG + i]);
            fac[threadIdx.x] = rsqrtf(t / SD + P.eps_gated);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < DI; i += NT)
            xs[(i >> 6) * CH + (i & 63)] = __ldcg(&s->attn[i]) * fac[i >> 7] * __ldg(L.ssm_norm + (i & 127)) * silu(__ldcg(&s->z[i]));
        __syncthreads();
        block_sums(xs, sums, DI);
        }
        if (RP) {
            unsigned a[18];
            int r = gw;
            if (r < H) ldrp(L.out, r, lane, a);
            for (; r < H; r += W) {
                unsigned b[18];
                const int rn = r + W;
                if (rn < H) ldrp(L.out, rn, lane, b);
                float acc = dot2v(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], xs + lane * CH, sums[lane]) +
                            dot2v(a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], xs + (lane + 32) * CH, sums[lane + 32]);
                acc = warp_sum(acc);
                if (lane == 0) P.out[r] = P.h[r] + acc;
#pragma unroll
                for (int j = 0; j < 18; ++j) a[j] = b[j];
            }
        } else {
        uint4 c[5];
        int r = gw;
        if (PFX) {
#pragma unroll
            for (int i = 0; i < 5; ++i) c[i] = cpf[i];   // row gw is staged in stg, c = row gw + W (if any)
        } else if (r < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)r * ROW_O);
#pragma unroll
            for (int i = 0; i < 4; ++i) c[i] = __ldg(p + lane + 32 * i);
            if (lane < 16) c[4] = __ldg(p + lane + 128); }
        for (; r < H; r += W) {
            if (!(PFX && r == gw)) {
#pragma unroll
            for (int i = 0; i < 4; ++i) stg4[lane + 32 * i] = c[i];
            if (lane < 16) stg4[lane + 128] = c[4];
            __syncwarp();
            const int rn = r + W;
            if (rn < H) { const uint4 * p = (const uint4 *)(L.out + (size_t)rn * ROW_O);
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] = __ldg(p + lane + 32 * i);
                if (lane < 16) c[4] = __ldg(p + lane + 128); }
            }
            float acc = dot2w(stg + lane * 9, xs + lane * CH, sums[lane]) + dot2w(stg + (lane + 32) * 9, xs + (lane + 32) * CH, sums[lane + 32]);
            acc = warp_sum(acc);
            if (lane == 0) P.out[r] = P.h[r] + acc;
            __syncwarp();
        }
        }
    }
}

// D1BF: dot2v for two activation sets sharing one weight decode; per set the exact dot2v fmaf chain and final expression
__device__ __forceinline__ void dot2v2(unsigned w0, unsigned w1, unsigned w2, unsigned w3, unsigned w4, unsigned w5, unsigned w6,
                                       unsigned w7, unsigned w8, const float * xc, float2 sx, const float * yc, float2 sy,
                                       float & rx, float & ry) {
    float d0 = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
    float d1 = __half2float(__ushort_as_half((unsigned short)(w4 >> 16)));
    unsigned q0[4] = { __byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432), __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432) };
    unsigned q1[4] = { w5, w6, w7, w8 };
    float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned lo = (q0[i] & 0x0F0F0F0Fu) | 0x80808080u, hi = ((q0[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        float4 xl = *(const float4 *)(xc + 4 * i), xh = *(const float4 *)(xc + 16 + 4 * i);
        a0 = fmaf(nib(lo, 0), xl.x, a0); a0 = fmaf(nib(lo, 1), xl.y, a0); a0 = fmaf(nib(lo, 2), xl.z, a0); a0 = fmaf(nib(lo, 3), xl.w, a0);
        a0 = fmaf(nib(hi, 0), xh.x, a0); a0 = fmaf(nib(hi, 1), xh.y, a0); a0 = fmaf(nib(hi, 2), xh.z, a0); a0 = fmaf(nib(hi, 3), xh.w, a0);
        xl = *(const float4 *)(yc + 4 * i); xh = *(const float4 *)(yc + 16 + 4 * i);
        b0 = fmaf(nib(lo, 0), xl.x, b0); b0 = fmaf(nib(lo, 1), xl.y, b0); b0 = fmaf(nib(lo, 2), xl.z, b0); b0 = fmaf(nib(lo, 3), xl.w, b0);
        b0 = fmaf(nib(hi, 0), xh.x, b0); b0 = fmaf(nib(hi, 1), xh.y, b0); b0 = fmaf(nib(hi, 2), xh.z, b0); b0 = fmaf(nib(hi, 3), xh.w, b0);
        lo = (q1[i] & 0x0F0F0F0Fu) | 0x80808080u; hi = ((q1[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        xl = *(const float4 *)(xc + 32 + 4 * i); xh = *(const float4 *)(xc + 48 + 4 * i);
        a1 = fmaf(nib(lo, 0), xl.x, a1); a1 = fmaf(nib(lo, 1), xl.y, a1); a1 = fmaf(nib(lo, 2), xl.z, a1); a1 = fmaf(nib(lo, 3), xl.w, a1);
        a1 = fmaf(nib(hi, 0), xh.x, a1); a1 = fmaf(nib(hi, 1), xh.y, a1); a1 = fmaf(nib(hi, 2), xh.z, a1); a1 = fmaf(nib(hi, 3), xh.w, a1);
        xl = *(const float4 *)(yc + 32 + 4 * i); xh = *(const float4 *)(yc + 48 + 4 * i);
        b1 = fmaf(nib(lo, 0), xl.x, b1); b1 = fmaf(nib(lo, 1), xl.y, b1); b1 = fmaf(nib(lo, 2), xl.z, b1); b1 = fmaf(nib(lo, 3), xl.w, b1);
        b1 = fmaf(nib(hi, 0), xh.x, b1); b1 = fmaf(nib(hi, 1), xh.y, b1); b1 = fmaf(nib(hi, 2), xh.z, b1); b1 = fmaf(nib(hi, 3), xh.w, b1);
    }
    rx = d0 * (128.f * a0 - 136.f * sx.x) + d1 * (128.f * a1 - 136.f * sx.y);
    ry = d0 * (128.f * b0 - 136.f * sy.x) + d1 * (128.f * b1 - 136.f * sy.y);
}
// D1BF (GGML_CUDA_OCCAMY_N2GDN=1, with N2SPLIT=1 on the REPACK path): two decode sequences in ONE launch of the
// gdn_fused<0,1,1,0,1> pipeline. Every K=2048 phase-A weight row and every packed ssm_out row is read once and dotted
// with both activation sets. Per sequence each value is the same expression in the same order as gdn_fused<0,1,1,0,1>:
//   A: xs/sums of sequence k live in the first half of the (K=4096-sized) xs region: k * XS_HALF; sums + 32 k.
//   B: the RQK+PCS item loop runs for sequence 0, then for sequence 1 (own Scratch, own state rows).
//   C: the ssm_out GEMV runs in two K halves so both sequences' inputs fit the n=1 smem (3 blocks/SM kept): pass 1
//      computes dot2v(chunk lane) for both sequences from words 0..8 and keeps words 9..11 in registers, pass 2 computes
//      dot2v(chunk lane + 32) from words 9..17 and adds: acc = dotA + dotB, then warp_sum, exactly as RP phase C.
// Rows per warp in phase C are register-held: needs grid * NW * N2R >= H (host checks).
struct Params2 { Params p[2]; };
constexpr int XS_HALF = H / 64 * CH;   // 2176 floats: one K=2048 activation set
constexpr int N2R = 4;                 // max ssm_out rows per warp
// GDN2 (T2 = 1, GGML_CUDA_OCCAMY_GDN_T2=1): the same launch for ONE sequence and TWO tokens (MTP1 verify, n_rs_seq = 1).
// p[0] = token 1 (h/out row 0, scratch, snapshot plane 1 = state after token 1), p[1] = token 2 (h/out row 1, scratch2,
// plane 0 = state after token 2); both read the p[0] source row. Phase A is unchanged (two activation sets). Phase B runs
// one RQK+PCS item loop: per channel conv1d over [c0,c1,c2,x1] and [c1,c2,x1,x2] (as the unfused concat/ssm_conv),
// per state tile one read, token 1 delta rule -> plane-1 store, token 2 continues from the registers -> plane-0 store.
// Conv state rows written: plane 1 = [c1,c2,x1], plane 0 = [c2,x1,x2] (v channels in B by their owner, q/k in C).
// PCS publish counts on scratch's hcnt only; the finishing block builds xg/sg for both tokens. Phase C is unchanged.
// Phase A only: recompute stock's 1024 logical RMS lanes in a 256-thread
// persistent CTA. Each logical lane accumulates two elements before the two
// XOR-tree reductions. Both token rows' reductions share this CTA barrier.
static __device__ __forceinline__ void q8a_prepare(const Params2 & pp, uint4 * packed, float2 * ds,
                                                   float * partial) {
    const int tid = threadIdx.x, lane = tid & 31;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float * h = pp.p[k].h;
#pragma unroll
        for (int i = tid; i < 1024; i += NT) {
            const float x = h[i], y = h[i + 1024];
            const float sq = warp_sum(fmaf(y, y, x*x));
            if (lane == 0) partial[k*32+(i >> 5)] = sq;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float inv = rsqrtf(warp_sum(partial[k*32+lane]) / H + pp.p[k].eps_attn);
        for (int i = tid; i < H; i += NT) {
            const float x = (inv * pp.p[k].h[i]) * __ldg(pp.p[0].L.attn_norm + i);
            const float amax = warp_reduce_max<32>(fabsf(x));
            const float sum = warp_sum(x);
            const float d = amax / 127.0f;
            const int q = amax == 0.f ? 0 : int(roundf(x/d));
            const int other = __shfl_down_sync(0xffffffffu, q, 2);
            const int block = i >> 5;
            if (lane == 0) ds[k * 64 + block] = __half22float2(make_half2(d, sum));
            if ((lane & 2) == 0) {
                // One uint4 per (token, paired block, packed-weight word, lane).
                // x/y carry low-nibble values from even/odd bytes, z/w high.
                const int j = (lane & 15) >> 2;
                const int field = ((lane >> 4) * 2) + (lane & 1);
                const int slot = ((k * 8 + (block & 1) * 4 + j) * 32 + (block >> 1)) * 4 + field;
                ((unsigned *)packed)[slot] = (unsigned(q) & 65535u) | (unsigned(other) << 16);
            }
        }
    }
    __syncthreads();
}

// Shared weights stay in their native 18-byte Q4_0 layout. An unsigned pair
// of nibbles times a signed pair of activations requires two sm_52 XMADs.
static __device__ __forceinline__ void q8a_dot(const unsigned * w, const uint4 * packed,
                                             const float2 * ds, const uint4 (&first)[8], int lane, float & v0, float & v1) {
    float accum0 = 0.f, accum1 = 0.f;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        int e0 = 0, o0 = 0, e1 = 0, o1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = b == 0 ? __byte_perm(w[j], w[j+1], 0x5432) : w[j+5];
            const unsigned mask = 0x000f000fu;
            const unsigned le = v & mask, lo = (v >> 8) & mask;
            const unsigned he = (v >> 4) & mask, ho = (v >> 12) & mask;
            const uint4 a = b == 0 ? first[j] : packed[(b*4+j)*32+lane], z = b == 0 ? first[4+j] : packed[(8+b*4+j)*32+lane];
            e0 = ggml_cuda_sm52_mad2(le, a.x, e0); e1 = ggml_cuda_sm52_mad2(le, z.x, e1);
            o0 = ggml_cuda_sm52_mad2(lo, a.y, o0); o1 = ggml_cuda_sm52_mad2(lo, z.y, o1);
            e0 = ggml_cuda_sm52_mad2(he, a.z, e0); e1 = ggml_cuda_sm52_mad2(he, z.z, e1);
            o0 = ggml_cuda_sm52_mad2(ho, a.w, o0); o1 = ggml_cuda_sm52_mad2(ho, z.w, o1);
        }
        const unsigned scale_bits = b == 0 ? w[0] : w[4] >> 16;
        const float d4 = __half2float(__ushort_as_half((unsigned short)scale_bits));
        const float2 s0 = ds[2*lane+b], s1 = ds[64+2*lane+b];
        accum0 += d4 * ((e0+o0) * s0.x - 8.f*s0.y);
        accum1 += d4 * ((e1+o1) * s1.x - 8.f*s1.y);
    }
    v0 = accum0;
    v1 = accum1;
}

// Q8C: single-chunk (2 Q4_0 blocks = 9 words) XMAD dot for both tokens against smem packed Q8_1
static __device__ __forceinline__ void q8c_dot_chunk(
    const unsigned (&w)[9], const uint4 * packed, const float2 * ds,
    int half, int lane, float & acc0, float & acc1) {
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        int e0 = 0, o0 = 0, e1 = 0, o1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned v = b == 0 ? __byte_perm(w[j], w[j + 1], 0x5432) : w[j + 5];
            const unsigned mask = 0x000f000fu;
            const unsigned le = v & mask, lo = (v >> 8) & mask;
            const unsigned he = (v >> 4) & mask, ho = (v >> 12) & mask;
            const int pk_idx = (half * 8 + b * 4 + j) * 32 + lane;
            const uint4 a = packed[pk_idx];
            const uint4 z = packed[512 + pk_idx];
            e0 = ggml_cuda_sm52_mad2(le, a.x, e0); e1 = ggml_cuda_sm52_mad2(le, z.x, e1);
            o0 = ggml_cuda_sm52_mad2(lo, a.y, o0); o1 = ggml_cuda_sm52_mad2(lo, z.y, o1);
            e0 = ggml_cuda_sm52_mad2(he, a.z, e0); e1 = ggml_cuda_sm52_mad2(he, z.z, e1);
            o0 = ggml_cuda_sm52_mad2(ho, a.w, o0); o1 = ggml_cuda_sm52_mad2(ho, z.w, o1);
        }
        const unsigned scale_bits = b == 0 ? w[0] : (w[4] >> 16);
        const float d4 = __half2float(__ushort_as_half((unsigned short)scale_bits));
        const float2 s0 = ds[half * 64 + 2 * lane + b];
        const float2 s1 = ds[128 + half * 64 + 2 * lane + b];
        acc0 += d4 * ((e0 + o0) * s0.x - 8.f * s0.y);
        acc1 += d4 * ((e1 + o1) * s1.x - 8.f * s1.y);
    }
}

template <int T2>
__global__ void __launch_bounds__(NT, 3) gdn_fused2(const Params2 PP) {
    extern __shared__ float smem[];
    __shared__ float qn[SD], kn[SD], vv[CGC], red[NW];
    __shared__ float qn2[T2 ? SD : 1], kn2[T2 ? SD : 1], vv2[T2 ? CGC : 1], red2[T2 ? NW : 1];
    __shared__ unsigned hc[NVH], fin[NVH + 1];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + DI / 64 * CH);
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + (threadIdx.x >> 5) * STW_RP; uint4 * stg4 = (uint4 *)stg;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    const Layer & L = PP.p[0].L;
    Scratch * const s0 = PP.p[0].s, * const s1 = PP.p[1].s;

    if constexpr (T2 >= 2) {   // ---- phase A ----
        uint4 * packed = reinterpret_cast<uint4 *>(smem);
        float2 * scales = reinterpret_cast<float2 *>(packed + 2*8*32);
        float * partial = reinterpret_cast<float *>(scales + 2*64);
        static_assert(2*8*32*4 + 2*64*2 + 2*32 <= SMEM_FLOATS, "phase A scratch");
        q8a_prepare(PP, packed, scales, partial);
        uint4 first[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) { first[j] = packed[j*32+lane]; first[4+j] = packed[(8+j)*32+lane]; }
        uint4 c0, c1, c2;
        int t = gw;
        if (t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        for (; t < NROWS_A; t += W) {
            stg4[lane] = c0; stg4[lane + 32] = c1; if (lane < 8) stg4[lane + 64] = c2;
            __syncwarp();
            const int tn = t + W;
            if (tn < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
            const unsigned * w = stg + lane * 9;
            float a0, a1;
            q8a_dot(w, packed, scales, first, lane, a0, a1);
            a0 = warp_sum(a0); a1 = warp_sum(a1);
            if (lane == 0) {
                float * d0 = t < NQKV ? &s0->qkv[t] : t < NQKV + DI ? &s0->z[t - NQKV] : &s0->ab[t - NQKV - DI];
                float * d1 = t < NQKV ? &s1->qkv[t] : t < NQKV + DI ? &s1->z[t - NQKV] : &s1->ab[t - NQKV - DI];
                *d0 = a0; *d1 = a1;
            }
            __syncwarp();
        }
    } else {   // ---- phase A ----
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const float * hk = PP.p[k].h; float * xk = xs + k * XS_HALF;
            float ss = 0.f;
            for (int i = threadIdx.x; i < H; i += NT) { float v = hk[i]; xk[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
            ss = warp_sum(ss);
            if (lane == 0) red[warp] = ss;
            __syncthreads();
            float tot = 0.f;
#pragma unroll
            for (int i = 0; i < NW; ++i) tot += red[i];
            const float sc = rsqrtf(tot / H + PP.p[k].eps_attn);
            for (int i = threadIdx.x; i < H; i += NT) xk[(i >> 6) * CH + (i & 63)] *= sc * __ldg(L.attn_norm + i);
            __syncthreads();
            block_sums(xk, sums + 32 * k, H);
        }
        const float * xc0 = xs + lane * CH, * xc1 = xs + XS_HALF + lane * CH;
        const float2 sx0 = sums[lane], sx1 = sums[32 + lane];
        uint4 c0, c1, c2;
        int t = gw;
        if (t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        for (; t < NROWS_A; t += W) {
            stg4[lane] = c0; stg4[lane + 32] = c1; if (lane < 8) stg4[lane + 64] = c2;
            __syncwarp();
            const int tn = t + W;
            if (tn < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
            const unsigned * w = stg + lane * 9;
            float a0, a1;
            dot2v2(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], xc0, sx0, xc1, sx1, a0, a1);
            a0 = warp_sum(a0); a1 = warp_sum(a1);
            if (lane == 0) {
                float * d0 = t < NQKV ? &s0->qkv[t] : t < NQKV + DI ? &s0->z[t - NQKV] : &s0->ab[t - NQKV - DI];
                float * d1 = t < NQKV ? &s1->qkv[t] : t < NQKV + DI ? &s1->z[t - NQKV] : &s1->ab[t - NQKV - DI];
                *d0 = a0; *d1 = a1;
            }
            __syncwarp();
        }
    }
    grid_barrier(s0);
    if constexpr (T2) {   // ---- phase B (RQK + PCS), GDN2: both tokens per item ----
        const Params & P = PP.p[0];
        const int row = __ldg(P.s_copy);
        const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
        const float * S_in    = P.ssm_base  + (size_t)row * P.ssm_stride;
        float * const co_1 = PP.p[0].conv_out, * const co_2 = PP.p[1].conv_out;
        float * const so_1 = PP.p[0].S_out,    * const so_2 = PP.p[1].S_out;
        if (threadIdx.x < NVH) { hc[threadIdx.x] = 0; if (threadIdx.x == 0) fin[NVH] = 0; }
        const int it_lo = (int)(((long long)blockIdx.x * (NVH * NCG)) / gridDim.x);
        const int it_hi = (int)(((long long)(blockIdx.x + 1) * (NVH * NCG)) / gridDim.x);
        int kh_c = -1, hv_c = -1;
        float qs = 0.f, ks = 0.f, qs2 = 0.f, ks2 = 0.f, gexp = 0.f, beta = 0.f, gexp2 = 0.f, beta2 = 0.f;
        for (int it = it_lo; it < it_hi; ++it) {
            const int hv = it / (2 * NCG) + NKH * ((it / NCG) & 1);
            const int cg = it % NCG, kh = hv % NKH;
            const bool need_qk = kh != kh_c;
            {
                const int t = threadIdx.x;
                if (need_qk) {
                const int c = t < SD ? kh * SD + t : H + kh * SD + (t - SD);
                const float * ci = conv_in + c * 3; const float * w = L.conv_w + c * 4;
                const float x1 = __ldcg(&s0->qkv[c]), x2 = __ldcg(&s1->qkv[c]);
                const float i0 = ci[0], i1 = ci[1], i2 = ci[2];
                float y  = i0 * w[0] + i1 * w[1] + i2 * w[2] + x1 * w[3];
                float y2 = i1 * w[0] + i2 * w[1] + x1 * w[2] + x2 * w[3];
                y = silu(y); y2 = silu(y2);
                const float sq = warp_sum(y * y), sq2 = warp_sum(y2 * y2);
                if (lane == 0) { red[warp] = sq; red2[warp] = sq2; }
                if (t < SD) { qn[t] = y; qn2[t] = y2; } else { kn[t - SD] = y; kn2[t - SD] = y2; }
                }
                if (t < CGC) {
                    const int cv = 2 * H + hv * SD + cg * CGC + t;   // unique owner: read all, then write both planes
                    const float * cvi = conv_in + cv * 3; const float * wv = L.conv_w + cv * 4;
                    const float xv1 = __ldcg(&s0->qkv[cv]), xv2 = __ldcg(&s1->qkv[cv]);
                    const float i0 = cvi[0], i1 = cvi[1], i2 = cvi[2];
                    const float yv  = i0 * wv[0] + i1 * wv[1] + i2 * wv[2] + xv1 * wv[3];
                    const float yv2 = i1 * wv[0] + i2 * wv[1] + xv1 * wv[2] + xv2 * wv[3];
                    vv[t] = silu(yv); vv2[t] = silu(yv2);
                    float * o1 = co_1 + cv * 3; o1[0] = i1; o1[1] = i2;  o1[2] = xv1;
                    float * o2 = co_2 + cv * 3; o2[0] = i2; o2[1] = xv1; o2[2] = xv2;
                }
            }
            __syncthreads();
            if (need_qk) {
            qs  = rsqrtf((red[0] + red[1] + red[2] + red[3]) / SD + P.eps_l2) * P.scale_l2;
            ks  = rsqrtf((red[4] + red[5] + red[6] + red[7]) / SD + P.eps_l2) * P.scale_l2;
            qs2 = rsqrtf((red2[0] + red2[1] + red2[2] + red2[3]) / SD + P.eps_l2) * P.scale_l2;
            ks2 = rsqrtf((red2[4] + red2[5] + red2[6] + red2[7]) / SD + P.eps_l2) * P.scale_l2;
            kh_c = kh;
            }
            if (hv != hv_c) {
            const float dtb = __ldg(L.dt_bias + hv), sa = __ldg(L.ssm_a + hv);
            const float al = __ldcg(&s0->ab[hv]) + dtb;
            const float sp = al > 20.f ? al : logf(1.f + expf(al));
            gexp = expf(sp * sa);
            beta = 1.f / (1.f + expf(-__ldcg(&s0->ab[NVH + hv])));
            const float al2 = __ldcg(&s1->ab[hv]) + dtb;
            const float sp2 = al2 > 20.f ? al2 : logf(1.f + expf(al2));
            gexp2 = expf(sp2 * sa);
            beta2 = 1.f / (1.f + expf(-__ldcg(&s1->ab[NVH + hv])));
            hv_c = hv;
            }
            const int col0 = cg * CGC + warp * 2;
            const size_t toff = ((size_t)hv * SD + col0) * SD;
            const float4 * si = (const float4 *)(S_in + toff) + lane;
            float4 * so1 = (float4 *)(so_1 + toff) + lane;
            float4 * so2 = (float4 *)(so_2 + toff) + lane;
            float4 st[2];
            st[0] = si[0]; st[1] = si[SD / 4];   // read the whole tile before any store (so2 may alias si)
            float ssq = 0.f, ssq2 = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float4 a = st[j];
                {   // token 1
                const float4 q4 = *(const float4 *)(qn + 4 * lane), k4r = *(const float4 *)(kn + 4 * lane);
                const float4 qv = make_float4(q4.x * qs, q4.y * qs, q4.z * qs, q4.w * qs);
                const float4 kv4 = make_float4(k4r.x * ks, k4r.y * ks, k4r.z * ks, k4r.w * ks);
                float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                float delta = (vv[warp * 2 + j] - gexp * kvd) * beta;
                a.x = gexp * a.x + kv4.x * delta; a.y = gexp * a.y + kv4.y * delta;
                a.z = gexp * a.z + kv4.z * delta; a.w = gexp * a.w + kv4.w * delta;
                float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                so1[j * SD / 4] = a;
                if (lane == 0) s0->attn[hv * SD + col0 + j] = att;
                ssq += att * att;
                }
                {   // token 2, from token 1's state
                const float4 q4 = *(const float4 *)(qn2 + 4 * lane), k4r = *(const float4 *)(kn2 + 4 * lane);
                const float4 qv = make_float4(q4.x * qs2, q4.y * qs2, q4.z * qs2, q4.w * qs2);
                const float4 kv4 = make_float4(k4r.x * ks2, k4r.y * ks2, k4r.z * ks2, k4r.w * ks2);
                float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                float delta = (vv2[warp * 2 + j] - gexp2 * kvd) * beta2;
                a.x = gexp2 * a.x + kv4.x * delta; a.y = gexp2 * a.y + kv4.y * delta;
                a.z = gexp2 * a.z + kv4.z * delta; a.w = gexp2 * a.w + kv4.w * delta;
                float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                so2[j * SD / 4] = a;
                if (lane == 0) s1->attn[hv * SD + col0 + j] = att;
                ssq2 += att * att;
                }
            }
            __syncthreads();
            if (lane == 0) { red[warp] = ssq; red2[warp] = ssq2; }
            __syncthreads();
            if (threadIdx.x == 0) {
                float t = 0.f, t2 = 0.f;
                for (int i = 0; i < NW; ++i) { t += red[i]; t2 += red2[i]; }
                s0->ssq[hv * NCG + cg] = t; s1->ssq[hv * NCG + cg] = t2; hc[hv]++;
            }
            __syncthreads();
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x < NVH && hc[threadIdx.x] != 0) {
            const unsigned c = hc[threadIdx.x];
            const unsigned old = atomicAdd(&s0->hcnt[threadIdx.x], c);
            if ((old % NCG) + c == NCG) { fin[atomicAdd(&fin[NVH], 1u)] = threadIdx.x; }
        }
        __syncthreads();
        const unsigned nf = fin[NVH];
        if (nf) {
            __threadfence();
            for (unsigned f = threadIdx.x >> 7; f < nf; f += NT / SD) {
                const int hv = fin[f], t = threadIdx.x & (SD - 1), i = hv * SD + t;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    Scratch * s = k ? s1 : s0;
                    float tt = 0.f;
#pragma unroll
                    for (int q = 0; q < NCG; ++q) tt += __ldcg(&s->ssq[hv * NCG + q]);
                    const float fh = rsqrtf(tt / SD + P.eps_gated);
                    const float v = __ldcg(&s->attn[i]) * fh * __ldg(L.ssm_norm + t) * silu(__ldcg(&s->z[i]));
                    if constexpr (T2 == 3) {
                        const float amax = warp_reduce_max<32>(fabsf(v));
                        const float bs = warp_sum(v);
                        const float d = amax / 127.0f;
                        const int q = amax == 0.f ? 0 : int(roundf(v / d));
                        const int other = __shfl_down_sync(0xffffffffu, q, 2);
                        const int block = i >> 5;
                        if (lane == 0) s->q8c_ds[block] = __half22float2(make_half2(d, bs));
                        if ((lane & 2) == 0) {
                            const int half = block >> 6;
                            const int lb = block & 63;
                            const int j = (lane & 15) >> 2;
                            const int field = ((lane >> 4) * 2) + (lane & 1);
                            const int slot = ((half * 8 + (lb & 1) * 4 + j) * 32 + (lb >> 1)) * 4 + field;
                            ((unsigned *)s->q8c_pk)[slot] = (unsigned(q) & 65535u) | (unsigned(other) << 16);
                        }
                    } else {
                        s->xg[i] = v;
                        const float bs = warp_sum(v);
                        if (lane == 0) s->sg[i >> 5] = bs;
                    }
                }
            }
        }
    } else
#pragma unroll
    for (int k = 0; k < 2; ++k) {   // ---- phase B (RQK + PCS), one sequence at a time ----
        const Params & P = PP.p[k];
        Scratch * s = P.s;
        const int row = __ldg(P.s_copy);
        const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
        const float * S_in    = P.ssm_base  + (size_t)row * P.ssm_stride;
        if (k) __syncthreads();   // sequence 0's fin/hc/red readers are done
        if (threadIdx.x < NVH) { hc[threadIdx.x] = 0; if (threadIdx.x == 0) fin[NVH] = 0; }
        const int it_lo = (int)(((long long)blockIdx.x * (NVH * NCG)) / gridDim.x);
        const int it_hi = (int)(((long long)(blockIdx.x + 1) * (NVH * NCG)) / gridDim.x);
        int kh_c = -1, hv_c = -1;
        float qs = 0.f, ks = 0.f, gexp = 0.f, beta = 0.f;
        for (int it = it_lo; it < it_hi; ++it) {
            const int hv = it / (2 * NCG) + NKH * ((it / NCG) & 1);
            const int cg = it % NCG, kh = hv % NKH;
            const bool need_qk = kh != kh_c;
            {
                const int t = threadIdx.x;
                if (need_qk) {
                const int c = t < SD ? kh * SD + t : H + kh * SD + (t - SD);
                const float * ci = conv_in + c * 3; const float * w = L.conv_w + c * 4;
                float xnew = __ldcg(&s->qkv[c]);
                float y = ci[0] * w[0] + ci[1] * w[1] + ci[2] * w[2] + xnew * w[3];
                y = silu(y);
                float sq = warp_sum(y * y);
                if (lane == 0) red[warp] = sq;
                if (t < SD) qn[t] = y; else kn[t - SD] = y;
                }
                if (t < CGC) {
                    const int cv = 2 * H + hv * SD + cg * CGC + t;
                    const float * cvi = conv_in + cv * 3; const float * wv = L.conv_w + cv * 4;
                    float xv = __ldcg(&s->qkv[cv]);
                    float c1 = cvi[1], c2 = cvi[2];
                    float yv = cvi[0] * wv[0] + c1 * wv[1] + c2 * wv[2] + xv * wv[3];
                    vv[t] = silu(yv);
                    float * co = P.conv_out + cv * 3; co[0] = c1; co[1] = c2; co[2] = xv;
                }
            }
            __syncthreads();
            if (need_qk) {
            qs = rsqrtf((red[0] + red[1] + red[2] + red[3]) / SD + P.eps_l2) * P.scale_l2;
            ks = rsqrtf((red[4] + red[5] + red[6] + red[7]) / SD + P.eps_l2) * P.scale_l2;
            kh_c = kh;
            }
            if (hv != hv_c) {
            const float al = __ldcg(&s->ab[hv]) + __ldg(L.dt_bias + hv);
            const float sp = al > 20.f ? al : logf(1.f + expf(al));
            gexp = expf(sp * __ldg(L.ssm_a + hv));
            beta = 1.f / (1.f + expf(-__ldcg(&s->ab[NVH + hv])));
            hv_c = hv;
            }
            const float4 q4 = *(const float4 *)(qn + 4 * lane), k4r = *(const float4 *)(kn + 4 * lane);
            const float4 qv = make_float4(q4.x * qs, q4.y * qs, q4.z * qs, q4.w * qs);
            const float4 kv4 = make_float4(k4r.x * ks, k4r.y * ks, k4r.z * ks, k4r.w * ks);
            const int col0 = cg * CGC + warp * 2;
            const float4 * si = (const float4 *)(S_in + ((size_t)hv * SD + col0) * SD) + lane;
            float4 * so = (float4 *)(P.S_out + ((size_t)hv * SD + col0) * SD) + lane;
            float4 st[2];
            st[0] = si[0]; st[1] = si[SD / 4];
            float ssq = 0.f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float4 a = st[j];
                float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                float delta = (vv[warp * 2 + j] - gexp * kvd) * beta;
                a.x = gexp * a.x + kv4.x * delta; a.y = gexp * a.y + kv4.y * delta;
                a.z = gexp * a.z + kv4.z * delta; a.w = gexp * a.w + kv4.w * delta;
                float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                so[j * SD / 4] = a;
                if (lane == 0) s->attn[hv * SD + col0 + j] = att;
                ssq += att * att;
            }
            __syncthreads();
            if (lane == 0) red[warp] = ssq;
            __syncthreads();
            if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < NW; ++i) t += red[i]; s->ssq[hv * NCG + cg] = t; hc[hv]++; }
            __syncthreads();
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x < NVH && hc[threadIdx.x] != 0) {
            const unsigned c = hc[threadIdx.x];
            const unsigned old = atomicAdd(&s->hcnt[threadIdx.x], c);
            if ((old % NCG) + c == NCG) { fin[atomicAdd(&fin[NVH], 1u)] = threadIdx.x; }
        }
        __syncthreads();
        const unsigned nf = fin[NVH];
        if (nf) {
            __threadfence();
            for (unsigned f = threadIdx.x >> 7; f < nf; f += NT / SD) {
                const int hv = fin[f], t = threadIdx.x & (SD - 1), i = hv * SD + t;
                float tt = 0.f;
#pragma unroll
                for (int q = 0; q < NCG; ++q) tt += __ldcg(&s->ssq[hv * NCG + q]);
                const float fh = rsqrtf(tt / SD + P.eps_gated);
                const float v = __ldcg(&s->attn[i]) * fh * __ldg(L.ssm_norm + t) * silu(__ldcg(&s->z[i]));
                s->xg[i] = v;
                const float bs = warp_sum(v);
                if (lane == 0) s->sg[i >> 5] = bs;
            }
        }
    }
    grid_barrier(s0);
    {   // ---- phase C ----
        if constexpr (T2) {   // q/k conv state of both snapshot planes from the one source row
            const Params & P = PP.p[0];
            const int row = __ldg(P.s_copy);
            const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
            float * const co_1 = PP.p[0].conv_out, * const co_2 = PP.p[1].conv_out;
            for (int c = blockIdx.x * NT + threadIdx.x; c < 2 * H; c += gridDim.x * NT) {
                const float * ci = conv_in + c * 3;
                const float a = ci[1], b = ci[2];   // read before the plane-0 store (may alias ci)
                const float x1 = __ldcg(&s0->qkv[c]), x2 = __ldcg(&s1->qkv[c]);
                float * o1 = co_1 + c * 3; o1[0] = a; o1[1] = b;  o1[2] = x1;
                float * o2 = co_2 + c * 3; o2[0] = b; o2[1] = x1; o2[2] = x2;
            }
        } else
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const Params & P = PP.p[k];
            const int row = __ldg(P.s_copy);
            const float * conv_in = P.conv_base + (size_t)row * P.conv_stride;
            for (int c = blockIdx.x * NT + threadIdx.x; c < 2 * H; c += gridDim.x * NT) {
                const float * ci = conv_in + c * 3; float * co = P.conv_out + c * 3;
                const float a = ci[1], b = ci[2];
                co[0] = a; co[1] = b; co[2] = __ldcg(&P.s->qkv[c]);
            }
        }
        if constexpr (T2 == 3) {
            // Single-pass Q8C over repacked ssm_out (no pass-2 re-read, no pa[] staging)
            uint4 * const c_pk = reinterpret_cast<uint4 *>(smem);         // 1024 uint4s = 16384 B
            float2 * const c_ds = reinterpret_cast<float2 *>(c_pk + 1024); // 256 float2s = 2048 B
            static_assert(1024 * 4 + 256 * 2 <= SMEM3_FLOATS, "Q8C phase C smem");
            const uint4 * const rp0 = (const uint4 *)(L.out + (size_t)gw * ROW_O);
            const size_t rstep = (size_t)W * (ROW_O / 16);
            uint4 c0, c1, c2;
            if (gw < H) { c0 = __ldg(rp0 + lane); c1 = __ldg(rp0 + 32 + lane); c2 = __ldg(rp0 + 64 + lane); }
            for (int i = threadIdx.x; i < 512; i += NT) {
                c_pk[i]       = __ldcg(&s0->q8c_pk[i]);
                c_pk[512 + i] = __ldcg(&s1->q8c_pk[i]);
            }
            if (threadIdx.x < 128) {
                c_ds[threadIdx.x]       = __ldcg(&s0->q8c_ds[threadIdx.x]);
                c_ds[128 + threadIdx.x] = __ldcg(&s1->q8c_ds[threadIdx.x]);
            }
            __syncthreads();
            const uint4 * p = rp0;
#pragma unroll 1
            for (int r = gw; r < H; r += W, p += rstep) {
                const uint4 c3 = __ldg(p + 96 + lane);
                const uint2 ct = __ldg((const uint2 *)(p + 128) + lane);
                float acc0 = 0.f, acc1 = 0.f;
                {
                    const unsigned w0[9] = { c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x };
                    q8c_dot_chunk(w0, c_pk, c_ds, 0, lane, acc0, acc1);
                }
                const unsigned b2y = c2.y, b2z = c2.z, b2w = c2.w;
                if (r + W < H) {
                    c0 = __ldg(p + rstep + lane);
                    c1 = __ldg(p + rstep + 32 + lane);
                    c2 = __ldg(p + rstep + 64 + lane);
                }
                {
                    const unsigned w1[9] = { b2y, b2z, b2w, c3.x, c3.y, c3.z, c3.w, ct.x, ct.y };
                    q8c_dot_chunk(w1, c_pk, c_ds, 1, lane, acc0, acc1);
                }
                acc0 = warp_sum(acc0);
                acc1 = warp_sum(acc1);
                if (lane == 0) {
                    PP.p[0].out[r] = PP.p[0].h[r] + acc0;
                    PP.p[1].out[r] = PP.p[1].h[r] + acc1;
                }
            }
        } else {
            // pass 1: chunks 0..31 (words 0..8 of each lane) of every row of this warp, both sequences; partials -> smem
            const uint4 * const rp0 = (const uint4 *)(L.out + (size_t)gw * ROW_O);
            const size_t rstep = (size_t)W * (ROW_O / 16);   // uint4 per W rows
            float * const pa = (float *)stg + lane;          // [ri][seq][lane] in the warp's phase-C-idle stage words
            uint4 c0, c1, c2;
            if (gw < H) { c0 = __ldg(rp0 + lane); c1 = __ldg(rp0 + 32 + lane); c2 = __ldg(rp0 + 64 + lane); }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const Scratch * s = PP.p[k].s;
                for (int i = 4 * threadIdx.x; i < H; i += 4 * NT)
                    *(float4 *)(xs + k * XS_HALF + (i >> 6) * CH + (i & 63)) = __ldcg((const float4 *)(s->xg + i));
            }
            if (threadIdx.x < 32) ((float4 *)(sums + 32 * (threadIdx.x >> 4)))[threadIdx.x & 15] = __ldcg((const float4 *)(((threadIdx.x >> 4) ? s1 : s0)->sg) + (threadIdx.x & 15));
            __syncthreads();
            {
                const uint4 * p = rp0;
#pragma unroll 1
                for (int r = gw, ri = 0; r < H; r += W, ++ri, p += rstep) {
                    const uint4 a0 = c0, a1 = c1, a2 = c2;
                    if (r + W < H)
                    { c0 = __ldg(p + rstep + lane); c1 = __ldg(p + rstep + 32 + lane); c2 = __ldg(p + rstep + 64 + lane); }
                    float v0, v1;
                    dot2v2(a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, xs + lane * CH, sums[lane], xs + XS_HALF + lane * CH, sums[32 + lane], v0, v1);
                    pa[ri * 64] = v0; pa[ri * 64 + 32] = v1;
                }
            }
            // pass 2: chunks 32..63 (words 9..17): word-group 2 is re-read (9..11), then 3 and the tail
            __syncthreads();   // every warp is done with the first-half inputs
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const Scratch * s = PP.p[k].s;
                for (int i = 4 * threadIdx.x; i < H; i += 4 * NT)
                    *(float4 *)(xs + k * XS_HALF + (i >> 6) * CH + (i & 63)) = __ldcg((const float4 *)(s->xg + H + i));
            }
            if (threadIdx.x < 32) ((float4 *)(sums + 32 * (threadIdx.x >> 4)))[threadIdx.x & 15] = __ldcg((const float4 *)(((threadIdx.x >> 4) ? s1 : s0)->sg) + 16 + (threadIdx.x & 15));
            __syncthreads();
            {
                const uint4 * p = rp0;
#pragma unroll 1
                for (int r = gw, ri = 0; r < H; r += W, ++ri, p += rstep) {
                    // no prefetch here: one extra row in flight costs 16 B of spills at the 80-register cap (3 blocks/SM)
                    const uint4 b2 = __ldg(p + 64 + lane), b3 = __ldg(p + 96 + lane); const uint2 bt = __ldg((const uint2 *)(p + 128) + lane);
                    float v0, v1;
                    dot2v2(b2.y, b2.z, b2.w, b3.x, b3.y, b3.z, b3.w, bt.x, bt.y, xs + lane * CH, sums[lane], xs + XS_HALF + lane * CH, sums[32 + lane], v0, v1);
                    float acc0 = pa[ri * 64] + v0;
                    float acc1 = pa[ri * 64 + 32] + v1;
                    acc0 = warp_sum(acc0);
                    acc1 = warp_sum(acc1);
                    if (lane == 0) { PP.p[0].out[r] = PP.p[0].h[r] + acc0; PP.p[1].out[r] = PP.p[1].h[r] + acc1; }
                }
            }
        }
    }
}

struct DevState {
    Scratch * scratch = nullptr;
    int grid = 0;
    int grid_rp = 0;   // GDN3 card B: grid of the REPACK variant (co-resident blocks only)
    Scratch * scratch2 = nullptr;   // D1BF: second sequence's scratch
    int grid2 = 0;                  // D1BF: grid of gdn_fused2 (co-resident, <= grid_rp)
    bool q8a = false;
    bool q8c = false;
    int gridT2 = 0;                 // GDN2: grid of gdn_fused2<1> (co-resident, <= grid_rp)
};
__global__ void rp_pack(const unsigned * __restrict__ src, unsigned * __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // H * 576 words
    if (i < H * (ROW_O / 4)) { const int r = i / (ROW_O / 4), q = i % (ROW_O / 4); dst[i] = src[r * (ROW_O / 4) + rp_src(q)]; }
}
static DevState g_dev[GGML_CUDA_MAX_DEVICES];
static std::mutex g_mtx;

static bool env_flag(const char * name) {
    const char * e = getenv(name);
    return e != nullptr && atoi(e) != 0;
}
static bool g_debug = env_flag("GGML_CUDA_OCCAMY_DEBUG");
static bool qk_reuse() { static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN_QKREUSE"); return on; }   // C10, default OFF
static bool pc_share() { static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN_PCSHARE"); return on; }   // C11, default OFF
// GDN1, default OFF; engaged only with QKREUSE=1 PCSHARE=1 and the fused single launch (not GDN_3L)
static bool pref() { static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN_PREF") && qk_reuse() && pc_share() && !env_flag("GGML_CUDA_OCCAMY_GDN_3L"); return on; }
// GDN3 card B, default OFF: packed ssm_out layout for phase C (QKREUSE+PCSHARE fused path only; PREF is not combined)
static bool repack() { static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN_REPACK") && qk_reuse() && pc_share() && !env_flag("GGML_CUDA_OCCAMY_GDN_3L"); return on; }
// D1BF, default OFF: one gdn_fused2 launch for the N2SPLIT two-sequence window (REPACK path only)
static bool n2gdn() { static const bool on = env_flag("GGML_CUDA_OCCAMY_N2GDN") && repack(); return on; }
// GDN2, default OFF: one gdn_fused2<1> launch for the two-token single-sequence MTP1 verify window (REPACK path only)
static bool t2gdn() { static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN_T2") && repack(); return on; }
static long g_launch_t2 = 0;    // GDN2 dispatch counter
static long g_launch_n2 = 0;    // D1BF dispatch counter
static long g_launch_pf = 0;    // GDN1 dispatch counter
static long g_launch_rp = 0;    // GDN3 card B dispatch counter
static long g_launch_rqk = 0;   // C10 dispatch counter (launches of the reuse variant)
static long g_launch_pcs = 0;   // C11 dispatch counter (launches of the shared phase-C variant)
static long g_engaged = 0, g_rejected_gdn = 0;   // windows fused / GDN-layer windows rejected (DEBUG counter)

static void reject(const char * why) {
    if (!g_debug) { return; }
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (seen.insert(why).second) {
        fprintf(stderr, "occamy-gdn: window rejected: %s\n", why);
    }
}

static DevState & dev_state(int device) {
    DevState & d = g_dev[device];
    if (d.scratch == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **)&d.scratch, sizeof(Scratch)));
        CUDA_CHECK(cudaMemset(d.scratch, 0, sizeof(Scratch)));
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_fused<0>, NT, SMEM2_FLOATS * sizeof(float)));
        const bool rqk = qk_reuse(), pcs = pc_share();
        if (rqk && pcs) {
            int per_sm_rp = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_rp, gdn_fused<0, 1, 1>, NT, SMEM2_FLOATS * sizeof(float)));
            per_sm = std::min(per_sm, per_sm_rp);
            if (pref()) {
                int per_sm_pf = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_pf, gdn_fused<0, 1, 1, 1>, NT, SMEM2_FLOATS * sizeof(float)));
                per_sm = std::min(per_sm, per_sm_pf);
            }
        } else if (rqk) {
            int per_sm_r = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_r, gdn_fused<0, 1, 0>, NT, SMEM2_FLOATS * sizeof(float)));
            per_sm = std::min(per_sm, per_sm_r);
        } else if (pcs) {
            int per_sm_p = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_p, gdn_fused<0, 0, 1>, NT, SMEM2_FLOATS * sizeof(float)));
            per_sm = std::min(per_sm, per_sm_p);
        }
        const int nsm = ggml_cuda_info().devices[device].nsm;
        d.grid = std::max(1, per_sm) * nsm;   // all blocks co-resident -> grid barrier is safe
        if (g_debug) {
            fprintf(stderr, "occamy-gdn: device %d: %d SMs x %d blocks/SM -> grid %d, smem %zu B, qk_reuse %d, pc_share %d, pref %d\n",
                    device, nsm, per_sm, d.grid, SMEM2_FLOATS * sizeof(float), (int) rqk, (int) pcs, (int) pref());
        }
        if (per_sm < 1) { d.grid = -1; }
        if (repack()) {
            // the spinning grid barrier needs every block co-resident: the grid is exactly the measured occupancy of this
            // compiled specialization at its own smem size, never more
            int per_sm_rp = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_rp, gdn_fused<0, 1, 1, 0, 1>, NT, SMEM3_FLOATS * sizeof(float)));
            d.grid_rp = per_sm_rp * nsm;
            if (const char * e = getenv("GGML_CUDA_OCCAMY_GDN_RP_GRID")) {
                const int cap = atoi(e);
                if (cap > 0 && d.grid_rp > cap) { d.grid_rp = cap; }
            }
            fprintf(stderr, "occamy-gdn: repack variant %d blocks/SM -> grid %d (smem %zu B dyn)\n", per_sm_rp, d.grid_rp, SMEM3_FLOATS * sizeof(float));
            if (n2gdn()) {
                int per_sm2 = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm2, gdn_fused2<0>, NT, SMEM3_FLOATS * sizeof(float)));
                d.grid2 = std::min(per_sm2 * nsm, d.grid_rp);
                if (d.grid2 * NW * N2R < H) { d.grid2 = 0; }   // phase C holds at most N2R rows per warp
                if (d.grid2 > 0) {
                    CUDA_CHECK(cudaMalloc((void **)&d.scratch2, sizeof(Scratch)));
                    CUDA_CHECK(cudaMemset(d.scratch2, 0, sizeof(Scratch)));
                }
                fprintf(stderr, "occamy-gdn: n2gdn variant %d blocks/SM -> grid %d (smem %zu B dyn)\n", per_sm2, d.grid2, SMEM3_FLOATS * sizeof(float));
            }
            if (t2gdn()) {
                int per_smt = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_smt, gdn_fused2<1>, NT, SMEM3_FLOATS * sizeof(float)));
                d.gridT2 = std::min(per_smt * nsm, d.grid_rp);
                if (d.gridT2 * NW * N2R < H) { d.gridT2 = 0; }   // phase C holds at most N2R rows per warp
                if (env_flag("GGML_CUDA_OCCAMY_GDN_T2_Q8A")) {
                    int q8_residency = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&q8_residency, gdn_fused2<2>, NT, SMEM3_FLOATS * sizeof(float)));
                    d.q8a = ggml_cuda_info().devices[device].cc == 520 && per_smt == 3 &&
                            q8_residency == per_smt && d.gridT2 == 72 && nsm == 24;
                    std::fprintf(stderr, "N82_ADMISSION original_blocks=%d candidate_blocks=%d grid=%d admitted=%d\n",
                                 per_smt, q8_residency, d.gridT2, int(d.q8a));
                }
                if (env_flag("GGML_CUDA_OCCAMY_GDN_T2_Q8C")) {
                    int q8c_residency = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&q8c_residency, gdn_fused2<3>, NT, SMEM3_FLOATS * sizeof(float)));
                    d.q8c = ggml_cuda_info().devices[device].cc == 520 && per_smt == 3 &&
                            q8c_residency == per_smt && d.gridT2 == 72 && nsm == 24;
                    std::fprintf(stderr, "N82C_ADMISSION original_blocks=%d candidate_blocks=%d grid=%d admitted=%d\n",
                                 per_smt, q8c_residency, d.gridT2, int(d.q8c));
                }
                if (d.gridT2 > 0 && d.scratch2 == nullptr) {
                    CUDA_CHECK(cudaMalloc((void **)&d.scratch2, sizeof(Scratch)));
                    CUDA_CHECK(cudaMemset(d.scratch2, 0, sizeof(Scratch)));
                }
                fprintf(stderr, "occamy-gdn: t2 variant %d blocks/SM -> grid %d (smem %zu B dyn)\n", per_smt, d.gridT2, SMEM3_FLOATS * sizeof(float));
            }
        }
        // GDN3 card A (measurement only): GGML_CUDA_OCCAMY_GDN_GRID=N caps the grid at N (fewer co-resident blocks stay safe)
        if (const char * e = getenv("GGML_CUDA_OCCAMY_GDN_GRID")) {
            const int cap = atoi(e);
            if (cap > 0 && d.grid > cap) { d.grid = cap; fprintf(stderr, "occamy-gdn: grid cap -> %d\n", d.grid); }
        }
    }
    return d;
}

} // namespace occamy_gdn

using namespace occamy_gdn;

bool ggml_cuda_occamy_gdn_enabled() {
    static const bool on = env_flag("GGML_CUDA_OCCAMY_GDN");
    return on;
}

int ggml_cuda_occamy_gdn_check_budget() {
    static int left = getenv("GGML_CUDA_OCCAMY_GDN_CHECK") ? atoi(getenv("GGML_CUDA_OCCAMY_GDN_CHECK")) : 0;
    if (left > 0) { return left--; }
    return 0;
}

static bool is_q4_mat(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_Q4_0 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}
static bool is_f32_leaf(const ggml_tensor * t, int64_t ne0, int64_t ne1) {
    return t && t->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 &&
           t->ne[3] == 1 && ggml_is_contiguous(t) && ((uintptr_t)t->data & 15) == 0;
}
// root of a view chain (ggml keeps view_src pointing at the non-view base and view_offs cumulative)
static const ggml_tensor * vroot(const ggml_tensor * t) { return t && t->view_src ? t->view_src : t; }
static bool on_device(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    return t->buffer != nullptr && t->buffer->buft == ggml_backend_cuda_buffer_type(ctx.device);
}

bool ggml_cuda_occamy_gdn_match_window(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, ggml_cuda_occamy_gdn_match & m) {
    const ggml_tensor * norm = cgraph->nodes[i];
    if (norm->op != GGML_OP_RMS_NORM) { return false; }
    const ggml_tensor * h = norm->src[0];
    const int rows=(int)ggml_nrows(h);
    if (h->type != GGML_TYPE_F32 || h->ne[0] != H || !ggml_is_contiguous(h)) { return false; }
    if (rows != 1 && rows != 2) { return false; }

    // window end: first ADD after i that takes h directly (the attention residual add)
    constexpr int MAXW = 192;
    int end = -1;
    bool has_gdn = false;
    const ggml_tensor * gdn_node = nullptr;
    const int lim = std::min(cgraph->n_nodes, i + MAXW);
    for (int j = i + 1; j < lim; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_GATED_DELTA_NET) { has_gdn = true; if (!gdn_node) { gdn_node = t; } }
        if (t->op == GGML_OP_ADD && (t->src[0] == h || t->src[1] == h)) { end = j; break; }
    }
    if (!has_gdn) { return false; }   // not a GDN layer: stay silent, the MoE matcher owns those windows
    // GDN2: two rows of one sequence (MTP1 verify, gdn K = 2) when GGML_CUDA_OCCAMY_GDN_T2=1; else two rows = two sequences
    int nt = 1;
    if (rows == 2 && t2gdn() && ggml_get_op_params_i32(gdn_node, 0) == 2 && gdn_node->src[2]->ne[2] == 2 && gdn_node->src[2]->ne[3] == 1) { nt = 2; }
    const int ns = rows / nt;
    if (ns!=1 && !(ns==2 && env_flag("GGML_CUDA_OCCAMY_GDN_STATE_DIRECT") && ggml_cuda_info().devices[ctx.device].cc==520 && ggml_cuda_gdn_direct_pair_supported())) { return false; }
    m.n_seqs=ns; m.nt=nt;
    static std::set<std::string> dumped;
    int cur = i;
    auto rej = [&](const char * why) {
        g_rejected_gdn++;
        reject(why);
        if (g_debug && dumped.insert(why).second) {
            const int e = end < 0 ? lim - 1 : end;
            fprintf(stderr, "occamy-gdn: first rejected GDN window %d..%d (%s at node %d):\n", i, e, why, cur);
            for (int j = i; j <= e; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                fprintf(stderr, "  [%d] %s %s %s ne=[%ld,%ld,%ld,%ld] view_offs=%zu\n", j, ggml_op_name(t->op),
                        t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : "", t->name,
                        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3], t->view_offs);
            }
        }
        return false;
    };
    if (end < 0) { return rej("no residual ADD within 192 nodes"); }
    const int n = end - i + 1;

    const ggml_tensor * win[MAXW];
    for (int j = 0; j < n; ++j) { win[j] = cgraph->nodes[i + j]; }
    std::sort(win, win + n);
    auto in_win = [&](const ggml_tensor * t) { return std::binary_search(win, win + n, t); };

    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return rej("window node flagged as output"); }
        switch (t->op) {
            case GGML_OP_RMS_NORM: case GGML_OP_MUL: case GGML_OP_MUL_MAT: case GGML_OP_ADD: case GGML_OP_UNARY:
            case GGML_OP_SCALE: case GGML_OP_CPY: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE: case GGML_OP_CONT: case GGML_OP_CONCAT: case GGML_OP_GET_ROWS: case GGML_OP_SSM_CONV:
            case GGML_OP_GATED_DELTA_NET:
                break;
            default: return rej(ggml_op_name(t->op));
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * s = t->src[k];
            if (s == nullptr || s == h || in_win(s)) { continue; }
            const ggml_tensor * r = vroot(s);
            if (r->op == GGML_OP_NONE) {
                if (!on_device(ctx, r)) { return rej("leaf input not in this device's plain CUDA buffer"); }
                continue;
            }
            if (in_win(r)) { continue; }   // view of a window member created outside the window order (not expected)
            return rej("external computed input inside window");
        }
    }
    // every node except the last is consumed only inside the window
    for (int j = 0; j < n - 1; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        int uses = 0;
        for (int q = 0; q < n; ++q) {
            const ggml_tensor * u = cgraph->nodes[i + q];
            for (int k = 0; k < GGML_MAX_SRC; ++k) { if (u->src[k] == t) { uses++; } }
        }
        if (uses != ggml_node_get_use_count(cgraph, i + j)) { return rej("intermediate used outside the window"); }
    }

    // ---- structure ----
    const ggml_tensor * xn = nullptr;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        if (t->op == GGML_OP_MUL && t->src[0] == norm && is_f32_leaf(t->src[1], H, 1)) { xn = t; m.norm_w = t->src[1]; }
    }
    if (!xn) { return rej("no rms_norm*w"); }

    const ggml_tensor * mm_qkv = nullptr, * mm_gate = nullptr, * mm_ab[2] = { nullptr, nullptr }, * mm_out = nullptr;
    const ggml_tensor * gr_conv = nullptr, * gr_ssm = nullptr, * conv = nullptr, * silu_conv = nullptr, * gdn = nullptr;
    const ggml_tensor * zero_conv_root = nullptr, * zero_ssm_root = nullptr;
    const ggml_tensor * concat = nullptr;   // conv_input = concat(conv_state[3,8192], qkv^T[1,8192]) along dim 0
    const ggml_tensor * add_dt = nullptr, * softplus = nullptr, * mul_a = nullptr, * sigm = nullptr, * silu_z = nullptr;
    const ggml_tensor * rms_q = nullptr, * rms_k = nullptr, * rms_g = nullptr, * sc_q = nullptr, * sc_k = nullptr;
    const ggml_tensor * mul_norm = nullptr, * mul_z = nullptr, * cpy_conv = nullptr, * cpy_ssm = nullptr, * cpy_conv1 = nullptr;
    int n_ab = 0;
    for (int j = 1; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[i + j];
        cur = i + j;
        switch (t->op) {
            case GGML_OP_MUL_MAT: {
                const ggml_tensor * w = t->src[0];
                if (vroot(t->src[1]) == xn && t->ne[1] == rows && t->ne[2] == 1) {
                    if (is_q4_mat(w, H, NQKV)) { mm_qkv = t; }
                    else if (is_q4_mat(w, H, DI)) { mm_gate = t; }
                    else if (is_q4_mat(w, H, NVH) && n_ab < 2) { mm_ab[n_ab++] = t; }
                    else { return rej("unexpected mul_mat on xn"); }
                } else if (is_q4_mat(w, DI, H) && t->ne[1] == nt && t->ne[2] == ns) { mm_out = t; }
                else { return rej("unexpected mul_mat"); }
            } break;
            case GGML_OP_GET_ROWS: {
                if (t->src[1]->type != GGML_TYPE_I32 || vroot(t->src[1])->op != GGML_OP_NONE) { return rej("get_rows ids not an I32 leaf"); }
                if (ggml_nelements(t->src[1]) != ns || !ggml_is_contiguous(t->src[1])) { return rej("n_rs != n_seqs or noncontiguous ids"); }
                if (vroot(t->src[0])->op != GGML_OP_NONE || t->src[0]->type != GGML_TYPE_F32) { return rej("get_rows source not a leaf"); }
                if (m.s_copy && m.s_copy != t->src[1]) { return rej("two different s_copy tensors"); }
                m.s_copy = t->src[1];
                if (t->ne[0] == CONV_N && t->ne[1] == ns) { gr_conv = t; m.conv_base = (const float *) t->src[0]->data; m.conv_stride = t->src[0]->nb[1] / 4; }
                else if (t->ne[0] == SSM_N && t->ne[1] == ns) { gr_ssm = t; m.ssm_base = (const float *) t->src[0]->data; m.ssm_stride = t->src[0]->nb[1] / 4; }
                else { return rej("get_rows shape"); }
            } break;
            case GGML_OP_CONCAT: {
                if (concat || !gr_conv || vroot(t->src[0]) != gr_conv || !mm_qkv || vroot(t->src[1]) != mm_qkv || t->ne[0] != 3 + nt || t->ne[1] != NQKV ||
                    t->ne[2] != ns || t->ne[3] != 1 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
                    return rej("concat not concat(conv_state, qkv^T)");
                }
                concat = t;
            } break;
            case GGML_OP_SSM_CONV: {
                if (t->src[2] != nullptr || !is_f32_leaf(t->src[1], 4, NQKV) || !concat || t->src[0] != concat) { return rej("ssm_conv shape"); }
                if (t->ne[0] != NQKV || t->ne[1] != nt || t->ne[2] != ns) { return rej("ssm_conv output shape"); }
                conv = t; m.conv_w = t->src[1];
            } break;
            case GGML_OP_UNARY: {
                const ggml_unary_op u = ggml_get_unary_op(t);
                if (u == GGML_UNARY_OP_SILU && t->src[0] == conv) { silu_conv = t; }
                else if (u == GGML_UNARY_OP_SILU && mm_gate && vroot(t->src[0]) == mm_gate) { silu_z = t; }
                else if (u == GGML_UNARY_OP_SOFTPLUS && t->src[0] == add_dt) { softplus = t; }
                else if (u == GGML_UNARY_OP_SIGMOID && n_ab == 2 && (vroot(t->src[0]) == mm_ab[0] || vroot(t->src[0]) == mm_ab[1])) {
                    sigm = t; m.wbeta = vroot(t->src[0])->src[0];
                }
                else { return rej("unexpected unary"); }
            } break;
            case GGML_OP_ADD: {
                if (t == cgraph->nodes[end]) { break; }
                const ggml_tensor * a = t->src[0], * b = t->src[1];
                if (!is_f32_leaf(b, NVH, 1) || n_ab == 0 || !(vroot(a) == mm_ab[0] || (n_ab == 2 && vroot(a) == mm_ab[1]))) { return rej("unexpected add"); }
                add_dt = t; m.dt_bias = b; m.walpha = vroot(a)->src[0];
            } break;
            case GGML_OP_MUL: {
                if (t == xn) { break; }
                if (t->src[0] == softplus && is_f32_leaf(t->src[1], NVH, 1)) { mul_a = t; m.ssm_a = t->src[1]; }
                else if (t->src[0] == rms_g && is_f32_leaf(t->src[1], SD, 1)) { mul_norm = t; m.ssm_norm = t->src[1]; }
                else if (mul_norm && silu_z && ((t->src[0] == mul_norm && vroot(t->src[1]) == silu_z) || (t->src[1] == mul_norm && vroot(t->src[0]) == silu_z))) { mul_z = t; }
                else { return rej("unexpected mul"); }
            } break;
            case GGML_OP_RMS_NORM: {
                float eps = 0.f; memcpy(&eps, t->op_params, sizeof(float));
                const ggml_tensor * src = t->src[0];
                if (silu_conv && vroot(src) == silu_conv && src->ne[0] == SD && src->ne[1] == NKH && src->ne[2] == nt && src->ne[3] == ns) {
                    if (src->view_offs == 0) { rms_q = t; m.eps_l2 = eps; }
                    else if (src->view_offs == (size_t) H * 4) { rms_k = t; if (eps != m.eps_l2) { return rej("q/k l2 eps differ"); } }
                    else { return rej("l2 rms_norm view offset"); }
                } else if (gdn && vroot(src) == gdn && src->view_offs == 0 && src->ne[0] == SD && src->ne[1] == NVH && src->ne[2] == nt && src->ne[3] == ns) {
                    rms_g = t; m.eps_gated = eps;
                } else { return rej("unexpected rms_norm"); }
            } break;
            case GGML_OP_SCALE: {
                if (ggml_nelements(t) == 0) { break; }   // build_rs state_zero (rs_zero < 0)
                float sc = 0.f, bias = 0.f; memcpy(&sc, t->op_params, sizeof(float)); memcpy(&bias, (const float *) t->op_params + 1, sizeof(float));
                if (bias != 0.f) { return rej("scale with bias"); }
                if (t->src[0] == rms_q) { sc_q = t; m.scale_l2 = sc; }
                else if (t->src[0] == rms_k) { sc_k = t; if (sc != m.scale_l2) { return rej("q/k l2 scale differ"); } }
                else if (sc == 0.f && t->view_src && vroot(t)->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
                         t->src[0]->view_src && t->data == t->src[0]->data) {
                    // build_rs state_zero (rs_zero >= 0): in-place scale by 0 of one state row; applied as a memset before the kernel
                    if (ggml_nelements(t) == CONV_N && !m.zero_conv) { m.zero_conv = (float *) t->data; zero_conv_root = vroot(t); }
                    else if (ggml_nelements(t) == SSM_N && !m.zero_ssm) { m.zero_ssm = (float *) t->data; zero_ssm_root = vroot(t); }
                    else { return rej("state_zero scale shape"); }
                }
                else { return rej("unexpected scale"); }
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                if (ggml_get_op_params_i32(t, 0) != nt) { return rej("gdn K != tokens per sequence"); }
                const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * g = t->src[3], * b = t->src[4], * st = t->src[5];
                if (!sc_q || !sc_k || !silu_conv || !mul_a || !sigm || !gr_ssm) { return rej("gdn inputs appear before their producers"); }
                if (vroot(q) != sc_q || vroot(k) != sc_k || vroot(v) != silu_conv || v->view_offs != (size_t) 2 * H * 4 ||
                    vroot(g) != mul_a || vroot(b) != sigm || vroot(st) != gr_ssm) { return rej("gdn wiring"); }
                if (v->ne[0] != SD || v->ne[1] != NVH || v->ne[2] != nt || v->ne[3] != ns || q->ne[1] != NKH || g->ne[0] != 1 ||
                    st->ne[0] != SD || st->ne[1] != SD || st->ne[2] != NVH || st->ne[3] != ns) { return rej("gdn shapes"); }
                gdn = t;
            } break;
            case GGML_OP_CPY: {
                if (ggml_nelements(t) == 0) { break; }   // build_rs states_extra (n_rs == n_seqs)
                const ggml_tensor * src = t->src[0];
                if (nt == 2) {   // GDN2: conv rows after token 1 (offset 1) / after token 2 (offset 2); one 2-plane state copy
                    const bool conv_dst = gr_conv && vroot(t) == vroot(gr_conv->src[0]) && ggml_nelements(t) == CONV_N && ggml_is_contiguous(t) && t->type == GGML_TYPE_F32;
                    if (concat && conv_dst && vroot(src) == concat && src->ne[0] == 3 && src->ne[1] == NQKV && src->ne[2] == 1 && src->view_offs == 4 && !cpy_conv1) {
                        cpy_conv1 = t; m.conv_out1 = (float *) t->data;
                    } else if (concat && conv_dst && vroot(src) == concat && src->ne[0] == 3 && src->ne[1] == NQKV && src->ne[2] == 1 && src->view_offs == 8 && !cpy_conv) {
                        cpy_conv = t; m.conv_out = (float *) t->data;
                    } else if (gdn && vroot(src) == gdn && src->view_offs == (size_t) SD * NVH * 2 * 4 && ggml_nelements(src) == SSM_N * 2 && ggml_is_contiguous(src) &&
                               gr_ssm && vroot(t) == vroot(gr_ssm->src[0]) && t->type == GGML_TYPE_F32 && t->ne[0] == SSM_N && t->ne[1] == 1 && t->ne[2] == 2 &&
                               t->ne[3] == 1 && t->nb[0] == 4 && !cpy_ssm) {
                        cpy_ssm = t; m.state_out = (float *) t->data; m.state_out1 = (float *) ((char *) t->data + t->nb[2]);
                    } else { return rej("unexpected cpy (t2)"); }
                } else
                if (concat && vroot(src) == concat && src->view_offs == 4 && src->ne[0] == 3 && src->ne[1] == NQKV && src->ne[2] == ns &&
                    gr_conv && vroot(t) == vroot(gr_conv->src[0]) && ggml_nelements(t) == CONV_N*ns && ggml_is_contiguous(t) && t->type == GGML_TYPE_F32) {
                    cpy_conv = t; m.conv_out = (float *) t->data;
                } else if (gdn && vroot(src) == gdn && src->view_offs == (size_t) SD * NVH * ns * 4 && ggml_nelements(src) == SSM_N*ns && ggml_is_contiguous(src) &&
                    gr_ssm && vroot(t) == vroot(gr_ssm->src[0]) && ggml_nelements(t) == SSM_N*ns && ggml_is_contiguous(t) && t->type == GGML_TYPE_F32) {
                    cpy_ssm = t; m.state_out = (float *) t->data;
                } else { return rej("unexpected cpy"); }
            } break;
            default: break;   // views / reshape / transpose / permute / cont / concat: checked through their consumers
        }
    }
    if (!mm_qkv || !mm_gate || n_ab != 2 || !mm_out || !gr_conv || !gr_ssm || !conv || !silu_conv || !gdn || !add_dt || !softplus ||
        !mul_a || !sigm || !silu_z || !rms_q || !rms_k || !rms_g || !sc_q || !sc_k || !mul_norm || !mul_z || !cpy_conv || !cpy_ssm) {
        return rej("missing GDN block component");
    }
    if ((zero_conv_root && zero_conv_root != vroot(gr_conv->src[0])) || (zero_ssm_root && zero_ssm_root != vroot(gr_ssm->src[0]))) {
        return rej("state_zero row not in the get_rows source");
    }
    if (m.walpha == m.wbeta) { return rej("alpha/beta resolve to the same weight"); }
    if (vroot(mm_out->src[1]) != mul_z) { return rej("ssm_out input is not the gated norm output"); }
    ggml_tensor * out = cgraph->nodes[end];
    if (!((vroot(out->src[0]) == mm_out && out->src[1] == h) || (vroot(out->src[1]) == mm_out && out->src[0] == h))) { return rej("residual add wiring"); }
    if (out->type != GGML_TYPE_F32 || out->ne[0] != H || ggml_nrows(out) != rows || !ggml_is_contiguous(out)) { return rej("output shape"); }
    if (nt == 2) {   // GDN2: every output plane is a whole row of its state buffer (no partial overlap with the source row)
        if (!cpy_conv1 || ((uintptr_t) m.conv_out1 & 15) || ((uintptr_t) m.state_out1 & 15)) { return rej("t2 snapshot outputs"); }
        const ptrdiff_t c0 = m.conv_out - m.conv_base, c1 = m.conv_out1 - m.conv_base, s0 = m.state_out - m.ssm_base, s1 = m.state_out1 - m.ssm_base;
        if (c0 < 0 || c1 < 0 || s0 < 0 || s1 < 0 || c0 % (ptrdiff_t) m.conv_stride || c1 % (ptrdiff_t) m.conv_stride ||
            s0 % (ptrdiff_t) m.ssm_stride || s1 % (ptrdiff_t) m.ssm_stride || c0 == c1 || s0 == s1) { return rej("t2 snapshot rows not row-aligned"); }
    }
    if (m.conv_stride < CONV_N || m.ssm_stride < SSM_N || ((uintptr_t) m.conv_base & 15) || ((uintptr_t) m.ssm_base & 15) ||
        ((uintptr_t) m.conv_out & 15) || ((uintptr_t) m.state_out & 15)) { return rej("state layout/alignment"); }

    if(ns==2 && (m.conv_stride!=CONV_N || m.ssm_stride!=SSM_N ||
        ((uintptr_t)m.conv_out-(uintptr_t)m.conv_base)%(CONV_N*sizeof(float)) ||
        ((uintptr_t)m.state_out-(uintptr_t)m.ssm_base)%(SSM_N*sizeof(float)))) { return rej("paired state rows must be contiguous and aligned"); }
    if(ns==2) {
        if(!ggml_are_same_stride(gdn->src[0],gdn->src[1]) || !ggml_is_contiguous_rows(gdn->src[0]) ||
           !ggml_is_contiguous_rows(gdn->src[2]) || !ggml_is_contiguous(gdn->src[3]) || !ggml_is_contiguous(gdn->src[4]) ||
           gdn->src[0]->ne[0]!=SD || gdn->src[0]->ne[3]!=2 || gdn->src[3]->ne[0]!=1) return rej("direct state input stride/shape");
        for(int j=0;j<n;++j){
            const ggml_tensor*t=cgraph->nodes[i+j];
            for(int k=0;k<GGML_MAX_SRC;++k){
                if(!t->src[k] || vroot(t->src[k])!=gr_ssm)continue;
                if((t==gdn && k==5) || t->op==GGML_OP_VIEW || t->op==GGML_OP_RESHAPE ||
                   (t->op==GGML_OP_CPY && ggml_nelements(t)==0))continue;
                return rej("direct state gather has another consumer");
            }
        }
        m.gr_ssm=gr_ssm;m.gdn=gdn;m.cpy_ssm=cpy_ssm;
    }
    memcpy(&m.eps_attn, norm->op_params, sizeof(float));
    m.n_nodes = n; m.h = h; m.out = out;
    m.wqkv = mm_qkv->src[0]; m.wgate = mm_gate->src[0]; m.wout = mm_out->src[0];
    for (const ggml_tensor * w : { m.wqkv, m.wgate, m.walpha, m.wbeta, m.wout, m.conv_w, m.dt_bias, m.ssm_a, m.ssm_norm, m.norm_w }) {
        if (!on_device(ctx, w)) { return rej("weight not in plain CUDA buffer"); }
    }
    DevState & d = dev_state(ctx.device);
    if (d.grid <= 0) { return rej("occupancy 0"); }
    if(nt==2 && g_debug) { static bool once=false;if(!once){fprintf(stderr,"occamy-gdn: matched GDN2 (MTP1 verify, 2 tokens) window\n");once=true;} }
    if(ns==2 && g_debug) { static bool once=false;if(!once){fprintf(stderr,"occamy-state-direct: matched two-sequence window, n_rs=2\n");once=true;} }
    g_engaged++;
    if (g_debug && ns==1 && (g_engaged == 1 || g_engaged % 3000 == 0)) {
        fprintf(stderr, "occamy-gdn: fused GDN window %d..%d (%d nodes) %s -> %s; engaged %ld, GDN windows rejected %ld\n",
                i, end, n, norm->name, out->name, g_engaged, g_rejected_gdn);
    }
    return true;
}

// GDN3 card B: per-device map ssm_out weight -> packed copy (made once; GGML_CUDA_OCCAMY_GDN_REPACK_LAYERS=N packs only the
// first N distinct layers seen, the rest keep the original layout and kernel). Packing runs on a private stream with a
// sync, so it is legal while the compute stream is being captured (relaxed mode); the graph only sees the launch.
static const uint8_t * rp_lookup(int device, const ggml_tensor * w, cudaStream_t st) {
    static std::map<const void *, void *> pk[GGML_CUDA_MAX_DEVICES];
    static int limit = getenv("GGML_CUDA_OCCAMY_GDN_REPACK_LAYERS") ? atoi(getenv("GGML_CUDA_OCCAMY_GDN_REPACK_LAYERS")) : 1 << 30;
    static int made[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = pk[device].find(w->data);
    if (it != pk[device].end()) { return (const uint8_t *) it->second; }
    if (made[device] >= limit) { pk[device][w->data] = nullptr; return nullptr; }
    (void) st;
    const size_t bytes = (size_t) H * ROW_O;
    size_t f0 = 0, f1 = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&f0, &tot));
    void * dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    cudaStream_t ps;
    CUDA_CHECK(cudaStreamCreateWithFlags(&ps, cudaStreamNonBlocking));
    rp_pack<<<(H * (ROW_O / 4) + 255) / 256, 256, 0, ps>>>((const unsigned *) w->data, (unsigned *) dst);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ps));
    CUDA_CHECK(cudaMemGetInfo(&f1, &tot));
    long bad = 0;
    if (g_debug) {   // byte-exact check on the host: every packed word equals its source word, and the map is a bijection
        std::vector<unsigned> a((size_t) H * ROW_O / 4), b(a.size());
        CUDA_CHECK(cudaMemcpyAsync(a.data(), w->data, bytes, cudaMemcpyDeviceToHost, ps));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), dst, bytes, cudaMemcpyDeviceToHost, ps));
        CUDA_CHECK(cudaStreamSynchronize(ps));
        std::vector<int> hit(ROW_O / 4, 0);
        for (int q = 0; q < ROW_O / 4; ++q) { hit[rp_src(q)]++; }
        for (int q = 0; q < ROW_O / 4; ++q) { if (hit[q] != 1) { bad++; } }
        for (int r = 0; r < H; ++r) for (int q = 0; q < ROW_O / 4; ++q) {
            if (b[(size_t) r * (ROW_O / 4) + q] != a[(size_t) r * (ROW_O / 4) + rp_src(q)]) { bad++; }
        }
        // inverse: unpack the packed row and compare it with the original bytes
        std::vector<unsigned> u(ROW_O / 4);
        for (int r = 0; r < H; ++r) {
            for (int q = 0; q < ROW_O / 4; ++q) { u[rp_src(q)] = b[(size_t) r * (ROW_O / 4) + q]; }
            if (memcmp(u.data(), a.data() + (size_t) r * (ROW_O / 4), ROW_O) != 0) { bad++; }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(ps));
    made[device]++;
    fprintf(stderr, "occamy-gdn: repacked ssm_out %s (%d/%d) %zu B, VRAM free %zu -> %zu MiB%s%ld\n", w->name, made[device],
            limit, bytes, f0 >> 20, f1 >> 20, g_debug ? ", unpack mismatches " : "", g_debug ? bad : 0L);
    if (bad) { CUDA_CHECK(cudaFree(dst)); pk[device][w->data] = nullptr; return nullptr; }
    pk[device][w->data] = dst;
    return (const uint8_t *) dst;
}

struct gdn_row_ov {   // D1b per-row override of the state rows and residual row
    int row; const int32_t * s_copy; const float * conv_base; const float * ssm_base; float * conv_out; float * S_out;
};
static void gdn_launch_impl(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m, float * out_override, const gdn_row_ov * ov);
static Params fill_params(const ggml_cuda_occamy_gdn_match & m, float * out_override, const gdn_row_ov * ov, Scratch * scratch);
void ggml_cuda_occamy_gdn_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m, float * out_override) {
    gdn_launch_impl(ctx, m, out_override, nullptr);
}

static uint64_t g_graph_gen = 1;
void ggml_cuda_occamy_gdn_graph_begin() { ++g_graph_gen; }

bool ggml_cuda_occamy_gdn_split2(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m) {
    if (m.n_seqs != 2 || m.conv_stride != (size_t) CONV_N || m.ssm_stride != (size_t) SSM_N) { return false; }
    const ptrdiff_t dc = m.conv_out - m.conv_base, ds = m.state_out - m.ssm_base;
    if (dc < 0 || ds < 0 || dc % CONV_N || ds % SSM_N || dc / CONV_N != ds / SSM_N) { return false; }
    const int64_t head = dc / CONV_N;
    cudaStream_t st = ctx.stream();
    // s_copy is a graph input uploaded before compute and shared by every GDN layer: read it once per graph
    static uint64_t gen = 0; static const void * ptr = nullptr; static int32_t sv[2] = { 0, 0 };
    if (gen != g_graph_gen || ptr != m.s_copy->data) {
        CUDA_CHECK(cudaMemcpyAsync(sv, m.s_copy->data, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        gen = g_graph_gen; ptr = m.s_copy->data;
    }
    if (sv[0] < 0 || sv[1] < 0) { return false; }
    static int32_t * dzero[GGML_CUDA_MAX_DEVICES] = {};
    if (!dzero[ctx.device]) {
        CUDA_CHECK(cudaMalloc(&dzero[ctx.device], sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(dzero[ctx.device], 0, sizeof(int32_t), st));
    }
    // build_rs state_zero rows first (once: a later memset could clobber a row the first launch already wrote)
    if (m.zero_conv) { CUDA_CHECK(cudaMemsetAsync(m.zero_conv, 0, CONV_N * sizeof(float), st)); }
    if (m.zero_ssm)  { CUDA_CHECK(cudaMemsetAsync(m.zero_ssm,  0, SSM_N  * sizeof(float), st)); }
    if (!(sv[0] == head && sv[1] == head + 1)) {
        // general gather (reorder, copy/fork [r,r], zeroed joiner): read every source row, then write the destination rows
        ggml_cuda_pool_alloc<float> tmp(ctx.pool(), 2 * (size_t) (CONV_N + SSM_N));
        float * tc = tmp.get(), * ts = tmp.get() + 2 * (size_t) CONV_N;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaMemcpyAsync(tc + (size_t) k * CONV_N, m.conv_base + (size_t) sv[k] * CONV_N, CONV_N * sizeof(float), cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(ts + (size_t) k * SSM_N,  m.ssm_base  + (size_t) sv[k] * SSM_N,  SSM_N  * sizeof(float), cudaMemcpyDeviceToDevice, st));
        }
        CUDA_CHECK(cudaMemcpyAsync(m.conv_out,  tc, 2 * CONV_N * sizeof(float), cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(m.state_out, ts, 2 * SSM_N  * sizeof(float), cudaMemcpyDeviceToDevice, st));
        if (g_debug) { static long n_g = 0; ++n_g; if ((n_g & (n_g - 1)) == 0) { fprintf(stderr, "occamy-gdn: N2SPLIT gather s_copy=[%d,%d] head=%ld (%ld)\n", sv[0], sv[1], (long) head, n_g); } }
    }
    DevState & d = dev_state(ctx.device);
    const uint8_t * pk = n2gdn() && d.grid2 > 0 ? rp_lookup(ctx.device, m.wout, st) : nullptr;
    Params2 PP;
    for (int k = 0; k < 2; ++k) {
        const gdn_row_ov ov = { k, dzero[ctx.device], m.conv_out + (size_t) k * CONV_N, m.state_out + (size_t) k * SSM_N,
                                m.conv_out + (size_t) k * CONV_N, m.state_out + (size_t) k * SSM_N };
        if (pk) { PP.p[k] = fill_params(m, nullptr, &ov, k ? d.scratch2 : d.scratch); PP.p[k].L.out = pk; }
        else    { gdn_launch_impl(ctx, m, nullptr, &ov); }
    }
    if (pk) {
        gdn_fused2<0><<<d.grid2, NT, SMEM3_FLOATS * sizeof(float), st>>>(PP);
        CUDA_CHECK(cudaGetLastError());
        if (g_debug && (++g_launch_n2 == 1 || g_launch_n2 % 3000 == 0)) { fprintf(stderr, "occamy-gdn: n2gdn launches %ld\n", g_launch_n2); }
    }
    if (g_debug) { static long n_s = 0; ++n_s; if ((n_s & (n_s - 1)) == 0) { fprintf(stderr, "occamy-gdn: N2SPLIT windows %ld\n", n_s); } }
    return true;
}

// Optional single-q2 measurement, covering exactly the handler's native launch.
struct N82Events {
    int count = 0, device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin[30], end[30];
    std::string names[30];
    bool candidate = false;
};
static thread_local N82Events n82_events;
static void n82_before(int device, cudaStream_t st, const char * name, int grid, bool candidate) {
    auto & e = n82_events;
    GGML_ASSERT(e.count < 30 && grid == 72);
    if (e.count == 0) {
        cudaStreamCaptureStatus capture;
        CUDA_CHECK(cudaStreamIsCapturing(st, &capture));
        GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
        e.stream = st; e.device = device; e.candidate = candidate;
        for (int i = 0; i < 30; ++i) {
            CUDA_CHECK(cudaEventCreate(&e.begin[i]));
            CUDA_CHECK(cudaEventCreate(&e.end[i]));
        }
    }
    GGML_ASSERT(e.stream == st && e.device == device && e.candidate == candidate);
    e.names[e.count] = name;
    CUDA_CHECK(cudaEventRecord(e.begin[e.count], st));
}
static void n82_after(cudaStream_t st) {
    auto & e = n82_events;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e.end[e.count], st));
    if (++e.count == 30) {
        CUDA_CHECK(cudaEventSynchronize(e.end[29]));
        for (int i = 0; i < 30; ++i) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e.begin[i], e.end[i]));
            std::fprintf(stderr, "N82_EVENT specialization=%d grid=72 slot=%d tensor=%s us=%.6f\n",
                         e.candidate ? 2 : 1, i, e.names[i].c_str(), ms*1000.f);
            CUDA_CHECK(cudaEventDestroy(e.begin[i]));
            CUDA_CHECK(cudaEventDestroy(e.end[i]));
        }
    }
}

// GDN2: MTP1 verify window (one sequence, two tokens) as one gdn_fused2<1> launch. p[0] = token 1 -> snapshot plane 1
// (conv_out1/state_out1), p[1] = token 2 -> plane 0 (conv_out/state_out); both read the s_copy source row.
static std::atomic<unsigned long long> c_gdn_q8c_calls{0};
static std::atomic<unsigned long long> c_gdn_q8a_calls{0};
static std::atomic<unsigned long long> c_gdn_fp_calls{0};
struct R3EngagedGdnAtExit {
    ~R3EngagedGdnAtExit() {
        const unsigned long long c = c_gdn_q8c_calls.load(std::memory_order_relaxed);
        const unsigned long long a = c_gdn_q8a_calls.load(std::memory_order_relaxed);
        const unsigned long long f = c_gdn_fp_calls.load(std::memory_order_relaxed);
        if (c > 0 || a > 0 || f > 0) {
            fprintf(stderr, "r3_engaged_gdn: q8c=%llu q8a=%llu fp32=%llu\n", c, a, f);
        }
    }
};
static R3EngagedGdnAtExit g_r3_gdn_atexit;

bool ggml_cuda_occamy_gdn_t2(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m) {
    if (m.nt != 2 || m.n_seqs != 1 || !m.conv_out1 || !m.state_out1) { return false; }
    DevState & d = dev_state(ctx.device);
    if (d.gridT2 <= 0 || d.scratch2 == nullptr) { return false; }
    cudaStream_t st = ctx.stream();
    const uint8_t * pk = rp_lookup(ctx.device, m.wout, st);
    if (!pk) { return false; }
    if (m.zero_conv) { CUDA_CHECK(cudaMemsetAsync(m.zero_conv, 0, CONV_N * sizeof(float), st)); }
    if (m.zero_ssm)  { CUDA_CHECK(cudaMemsetAsync(m.zero_ssm,  0, SSM_N  * sizeof(float), st)); }
    const int32_t * sc = (const int32_t *) m.s_copy->data;
    const gdn_row_ov ov0 = { 0, sc, m.conv_base, m.ssm_base, m.conv_out1, m.state_out1 };
    const gdn_row_ov ov1 = { 1, sc, m.conv_base, m.ssm_base, m.conv_out,  m.state_out  };
    Params2 PP;
    PP.p[0] = fill_params(m, nullptr, &ov0, d.scratch);  PP.p[0].L.out = pk;
    PP.p[1] = fill_params(m, nullptr, &ov1, d.scratch2); PP.p[1].L.out = pk;
    static const bool events = env_flag("GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS");
    if (events) n82_before(ctx.device, st, m.wqkv->name, d.gridT2, d.q8a || d.q8c);
    if (d.q8c) {
        gdn_fused2<3><<<d.gridT2, NT, SMEM3_FLOATS * sizeof(float), st>>>(PP);
        c_gdn_q8c_calls.fetch_add(1, std::memory_order_relaxed);
    } else if (d.q8a) {
        gdn_fused2<2><<<d.gridT2, NT, SMEM3_FLOATS * sizeof(float), st>>>(PP);
        c_gdn_q8a_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
        gdn_fused2<1><<<d.gridT2, NT, SMEM3_FLOATS * sizeof(float), st>>>(PP);
        c_gdn_fp_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (events) n82_after(st);
    CUDA_CHECK(cudaGetLastError());
    if (g_debug && (++g_launch_t2 == 1 || g_launch_t2 % 3000 == 0)) { fprintf(stderr, "occamy-gdn: t2 launches %ld\n", g_launch_t2); }
    return true;
}

static Params fill_params(const ggml_cuda_occamy_gdn_match & m, float * out_override, const gdn_row_ov * ov, Scratch * scratch) {
    Params P;
    P.L.attn_norm = (const float *) m.norm_w->data;
    P.L.conv_w    = (const float *) m.conv_w->data;
    P.L.ssm_a     = (const float *) m.ssm_a->data;
    P.L.dt_bias   = (const float *) m.dt_bias->data;
    P.L.ssm_norm  = (const float *) m.ssm_norm->data;
    P.L.qkv   = (const uint8_t *) m.wqkv->data;
    P.L.gate  = (const uint8_t *) m.wgate->data;
    P.L.alpha = (const uint8_t *) m.walpha->data;
    P.L.beta  = (const uint8_t *) m.wbeta->data;
    P.L.out   = (const uint8_t *) m.wout->data;
    P.h   = (const float *) m.h->data;
    P.out = out_override ? out_override : (float *) m.out->data;
    P.s_copy = (const int32_t *) m.s_copy->data;
    P.conv_base = m.conv_base; P.conv_stride = m.conv_stride; P.conv_out = m.conv_out;
    P.ssm_base  = m.ssm_base;  P.ssm_stride  = m.ssm_stride;  P.S_out    = m.state_out;
    if (ov) {
        P.h += (size_t) ov->row * H; P.out += (size_t) ov->row * H; P.s_copy = ov->s_copy;
        P.conv_base = ov->conv_base; P.conv_out = ov->conv_out; P.ssm_base = ov->ssm_base; P.S_out = ov->S_out;
    }
    P.s = scratch;
    P.eps_attn = m.eps_attn; P.eps_l2 = m.eps_l2; P.scale_l2 = m.scale_l2; P.eps_gated = m.eps_gated;
    return P;
}
static void gdn_launch_impl(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_gdn_match & m, float * out_override, const gdn_row_ov * ov) {
    static const bool three = env_flag("GGML_CUDA_OCCAMY_GDN_3L");
    DevState & d = dev_state(ctx.device);
    Params P = fill_params(m, out_override, ov, d.scratch);
    cudaStream_t st = ctx.stream();
    if (m.zero_conv && !ov) { CUDA_CHECK(cudaMemsetAsync(m.zero_conv, 0, CONV_N * sizeof(float), st)); }
    if (m.zero_ssm && !ov)  { CUDA_CHECK(cudaMemsetAsync(m.zero_ssm,  0, SSM_N  * sizeof(float), st)); }
    const size_t smem = SMEM2_FLOATS * sizeof(float);
    const bool rqk = qk_reuse();
    if (!three && repack() && d.grid_rp > 0) {
        if (const uint8_t * pk = rp_lookup(ctx.device, m.wout, st)) {
            P.L.out = pk;
            gdn_fused<0, 1, 1, 0, 1><<<d.grid_rp, NT, SMEM3_FLOATS * sizeof(float), st>>>(P);
            if (g_debug && (++g_launch_rp == 1 || g_launch_rp % 3000 == 0)) { fprintf(stderr, "occamy-gdn: repack launches %ld\n", g_launch_rp); }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const bool pcs = pc_share();
    if (three) {
        gdn_fused<1><<<d.grid, NT, smem, st>>>(P);
        if (rqk && pcs) {
            gdn_fused<2, 1, 1><<<d.grid, NT, smem, st>>>(P);
            gdn_fused<3, 0, 1><<<d.grid, NT, smem, st>>>(P);
        } else if (rqk) {
            gdn_fused<2, 1, 0><<<d.grid, NT, smem, st>>>(P);
            gdn_fused<3, 0, 0><<<d.grid, NT, smem, st>>>(P);
        } else if (pcs) {
            gdn_fused<2, 0, 1><<<d.grid, NT, smem, st>>>(P);
            gdn_fused<3, 0, 1><<<d.grid, NT, smem, st>>>(P);
        } else {
            gdn_fused<2, 0, 0><<<d.grid, NT, smem, st>>>(P);
            gdn_fused<3, 0, 0><<<d.grid, NT, smem, st>>>(P);
        }
    } else {
        if (rqk && pcs && pref()) {
            gdn_fused<0, 1, 1, 1><<<d.grid, NT, smem, st>>>(P);
            if (g_debug && (++g_launch_pf == 1 || g_launch_pf % 3000 == 0)) {
                fprintf(stderr, "occamy-gdn: pref launches %ld\n", g_launch_pf);
            }
        } else if (rqk && pcs) {
            gdn_fused<0, 1, 1><<<d.grid, NT, smem, st>>>(P);
        } else if (rqk) {
            gdn_fused<0, 1, 0><<<d.grid, NT, smem, st>>>(P);
        } else if (pcs) {
            gdn_fused<0, 0, 1><<<d.grid, NT, smem, st>>>(P);
        } else {
            gdn_fused<0, 0, 0><<<d.grid, NT, smem, st>>>(P);
        }
    }
    if (rqk && g_debug && (++g_launch_rqk == 1 || g_launch_rqk % 3000 == 0)) {
        fprintf(stderr, "occamy-gdn: qk_reuse launches %ld\n", g_launch_rqk);
    }
    if (pcs && g_debug && (++g_launch_pcs == 1 || g_launch_pcs % 3000 == 0)) {
        fprintf(stderr, "occamy-gdn: pc_share launches %ld\n", g_launch_pcs);
    }
    CUDA_CHECK(cudaGetLastError());
}
