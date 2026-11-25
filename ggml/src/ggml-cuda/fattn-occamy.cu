// Track F (native-occamy, 12 Oct 2026): GQA-aware split-K flash-attention DECODE kernel for sm_52.
// Shape it is built for: D = 256, 8 q-heads per kv-head, f16 K/V, f32 Q, one query token per stream.
// Env gate (default off): GGML_CUDA_OCCAMY_FA=1 (or GGML_CUDA_NATIVE_FA_SM52=1) -> use this kernel when applicable;
//                         GGML_CUDA_OCCAMY_FA=2 -> run this kernel AND the stock kernel, log max abs/rel diff, keep stock output.
// Knobs: GGML_CUDA_OCCAMY_FA_WAVES (blocks = waves * resident blocks, default 1).
//        GGML_CUDA_OCCAMY_FA_PIPE (E07, default 0 = off): 1 -> register-prefetch the next unit's K and V rows
//        while the current unit is computed; 2 -> prefetch K only (V still issued at step start).
//        Same split/warp/unit assignment and same arithmetic -> bitwise-identical output; only load timing moves.
//        3 -> head-split kernel: 8 warps/block, warp (u-lane w, head-group g) owns q-heads [4g,4g+4) of the
//        units the flag-off warp w owns (both head groups re-read the same K/V rows, mostly L1/L2 hits).
//        Half the q/o registers -> 2x resident warps. The per-value lane reduction tree (xor 16,8,4,2,1),
//        per-token softmax order and warp combine order are unchanged -> bitwise-identical output.
//        4 -> head-split + K prefetch (as 2).
// Design: one block = (kv-head, stream, KV split); 4 warps; each warp streams 4 KV tokens per step,
// lane l owns dims [8l, 8l+8) of a 256-dim row (one 16-B LDG per row), all 8 q-heads of that kv-head live
// in registers (64 fp32) and so does the output accumulator (64 fp32). K/V are read exactly once per block,
// converted f16 -> f32 (no half math). QK partials (4 tokens x 8 heads = 32 per lane) are reduce-scattered
// across the warp so lane l ends with score (token l>>3, head l&7). Partials (m, l, o) per block -> combine.
#include "common.cuh"
#include "fattn-occamy.cuh"
#include "fattn-occamy-kvq8.cuh"

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <atomic>

#define OFA_D   256
#define OFA_G   8
#define OFA_NW  4
#define OFA_T   4
#define OFA_NT  (OFA_NW*32)

static __device__ __forceinline__ void ofa_cvt8(const uint4 w, float * f) {
    const half2 * h = (const half2 *) &w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(h[i]);
        f[2*i + 0] = t.x;
        f[2*i + 1] = t.y;
    }
}

template <int n>
static __device__ __forceinline__ void ofa_rs_step(float * s, const int lane) {
    const bool up = lane & n;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const float send = up ? s[i] : s[i + n];
        const float keep = up ? s[i + n] : s[i];
        s[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, n, 32);
    }
}

struct ofa_params {
    const char * Q; const char * K; const char * V; const char * mask;
    float  * part_o;   // [nblk][8][256]
    float2 * part_ml;  // [nblk][8]
    int n_kv, nsplit, nkvh, mask_ne1, mask_ne3;
    float scale_log2;  // scale * log2(e)
    size_t nbq1, nbq2, nbq3;
    size_t nbk1, nbk2, nbk3;
    size_t nbv1, nbv2, nbv3;
    size_t nbm1, nbm3;
};

template <int PIPE>
__global__ void __launch_bounds__(OFA_NT, 2) ofa_decode_kernel(const ofa_params p) {
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;

    __shared__ float  s_p[OFA_NW][32];
    __shared__ float  s_a[OFA_NW][OFA_G];
    __shared__ float  s_o[OFA_NW][OFA_G][OFA_D];
    __shared__ float  s_m[OFA_NW][OFA_G];
    __shared__ float  s_l[OFA_NW][OFA_G];

    // q for the 8 heads, my 8 dims, pre-scaled by scale*log2(e)
    float q[OFA_G][8];
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(kvh*OFA_G + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_G][8];
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    const int   hl = lane & 7;       // head whose score this lane owns after the reduce-scatter
    const int   jl = lane >> 3;      // token (within the step) whose score this lane owns
    float m_run = -FLT_MAX/2.0f;     // running max for head hl (log2 domain), same on the 4 lanes of a head
    float l_run = 0.0f;              // per-lane partial of the softmax denominator for head hl

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    uint4 kn[OFA_T], vn[OFA_T];   // PIPE: rows of the next unit, loaded one step ahead
    if constexpr (PIPE != 0) {
        const int u = u0 + warp;
        if (u < u1) {
            const int t0 = u*OFA_T;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kn[j] = __ldg((const uint4 *) (Kb + (size_t)(t0 + j)*p.nbk1));
            if constexpr (PIPE == 1) {
#pragma unroll
                for (int j = 0; j < OFA_T; ++j) vn[j] = __ldg((const uint4 *) (Vb + (size_t)(t0 + j)*p.nbv1));
            }
        }
    }
    for (int u = u0 + warp; u < u1; u += OFA_NW) {
        const int t0 = u*OFA_T;
        uint4 kr[OFA_T], vr[OFA_T];
        if constexpr (PIPE == 0) {
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kr[j] = __ldg((const uint4 *) (Kb + (size_t)(t0 + j)*p.nbk1));
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) vr[j] = __ldg((const uint4 *) (Vb + (size_t)(t0 + j)*p.nbv1));
        } else {
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kr[j] = kn[j];
            if constexpr (PIPE == 1) {
#pragma unroll
                for (int j = 0; j < OFA_T; ++j) vr[j] = vn[j];
            } else {
#pragma unroll
                for (int j = 0; j < OFA_T; ++j) vr[j] = __ldg((const uint4 *) (Vb + (size_t)(t0 + j)*p.nbv1));
            }
            const int un = u + OFA_NW;
            if (un < u1) {
                const int tn = un*OFA_T;
#pragma unroll
                for (int j = 0; j < OFA_T; ++j) kn[j] = __ldg((const uint4 *) (Kb + (size_t)(tn + j)*p.nbk1));
                if constexpr (PIPE == 1) {
#pragma unroll
                    for (int j = 0; j < OFA_T; ++j) vn[j] = __ldg((const uint4 *) (Vb + (size_t)(tn + j)*p.nbv1));
                }
            }
        }
        const float mk = __half2float(Mb[t0 + jl]);

        float s[OFA_T*OFA_G];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < OFA_G; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*OFA_G + h] = acc;
            }
        }
        // reduce-scatter 32 values over 32 lanes: lane l ends with the full sum of value index l
        ofa_rs_step<16>(s, lane);
        ofa_rs_step< 8>(s, lane);
        ofa_rs_step< 4>(s, lane);
        ofa_rs_step< 2>(s, lane);
        ofa_rs_step< 1>(s, lane);
        const float sc = s[0] + mk*1.4426950408889634f;   // mask is -inf for masked cells
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        s_p[warp][lane] = pr;
        if (lane < OFA_G) s_a[warp][lane] = alpha;
        __syncwarp();
        float a[OFA_G];
        {
            const float4 a0 = *(const float4 *) &s_a[warp][0];
            const float4 a1 = *(const float4 *) &s_a[warp][4];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w; a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
        }
#pragma unroll
        for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[warp][j*OFA_G + 0];
            const float4 p1 = *(const float4 *) &s_p[warp][j*OFA_G + 4];
            const float pj[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int h = 0; h < OFA_G; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    // warp -> smem
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < OFA_G) { s_m[warp][lane] = m_run; s_l[warp][lane] = l_run; }
#pragma unroll
    for (int h = 0; h < OFA_G; ++h) {
        float4 * dp = (float4 *) &s_o[warp][h][8*lane];
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
    __syncthreads();

    // block combine across warps -> one partial per block
    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*OFA_G*OFA_D;
    for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += OFA_NT) {
        const int h = idx / OFA_D, d = idx % OFA_D;
        float M = s_m[0][h];
#pragma unroll
        for (int w = 1; w < OFA_NW; ++w) M = fmaxf(M, s_m[w][h]);
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < OFA_NW; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][h] - M);
        po[idx] = acc;
        if (d == 0) {
            float L = 0.0f;
#pragma unroll
            for (int w = 0; w < OFA_NW; ++w) L += s_l[w][h]*exp2f(s_m[w][h] - M);
            p.part_ml[(size_t) blk*OFA_G + h] = make_float2(M, L);
        }
    }
}

template <int n, int x>
static __device__ __forceinline__ void ofa_rs_step_x(float * s, const int lane) {
    const bool up = lane & x;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const float send = up ? s[i] : s[i + n];
        const float keep = up ? s[i + n] : s[i];
        s[i] = keep + __shfl_xor_sync(0xFFFFFFFF, send, x, 32);
    }
}

#define OFA_HG 4   // q-heads per warp in the head-split kernel

template <bool PFK>
__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_kernel(const ofa_params p) {
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int warp  = wid & (OFA_NW - 1);   // unit lane, same role as the flag-off warp index
    const int hg    = wid / OFA_NW;         // head group
    const int lane  = threadIdx.x & 31;

    __shared__ __align__(16) float s_p[2*OFA_NW][OFA_T*OFA_HG];
    __shared__ __align__(16) float s_a[2*OFA_NW][OFA_HG];
    __shared__ float  s_o[OFA_NW][OFA_G][OFA_D];
    __shared__ float  s_m[OFA_NW][OFA_G];
    __shared__ float  s_l[OFA_NW][OFA_G];

    float q[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(kvh*OFA_G + hg*OFA_HG + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    // after the reduce-scatter lane l holds value index l>>1 = token*4 + head (lanes l, l^1 duplicate)
    const int   jl = lane >> 3;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    uint4 kn[OFA_T];
    if constexpr (PFK) {
        const int u = u0 + warp;
        if (u < u1) {
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kn[j] = __ldg((const uint4 *) (Kb + (size_t)(u*OFA_T + j)*p.nbk1));
        }
    }
    for (int u = u0 + warp; u < u1; u += OFA_NW) {
        const int t0 = u*OFA_T;
        uint4 kr[OFA_T], vr[OFA_T];
        if constexpr (PFK) {
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kr[j] = kn[j];
        } else {
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kr[j] = __ldg((const uint4 *) (Kb + (size_t)(t0 + j)*p.nbk1));
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) vr[j] = __ldg((const uint4 *) (Vb + (size_t)(t0 + j)*p.nbv1));
        if constexpr (PFK) {
            const int un = u + OFA_NW;
            if (un < u1) {
#pragma unroll
                for (int j = 0; j < OFA_T; ++j) kn[j] = __ldg((const uint4 *) (Kb + (size_t)(un*OFA_T + j)*p.nbk1));
            }
        }
        const float mk = __half2float(Mb[t0 + jl]);

        float s[OFA_T*OFA_HG];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*OFA_HG + h] = acc;
            }
        }
        // same lane pairing order as the flag-off kernel (xor 16, 8, 4, 2, 1)
        ofa_rs_step_x<8, 16>(s, lane);
        ofa_rs_step_x<4,  8>(s, lane);
        ofa_rs_step_x<2,  4>(s, lane);
        ofa_rs_step_x<1,  2>(s, lane);
        s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
        const float sc = s[0] + mk*1.4426950408889634f;
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
        if (lane < 2*OFA_HG && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
        __syncwarp();
        float a[OFA_HG];
        {
            const float4 a0 = *(const float4 *) &s_a[wid][0];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
        }
#pragma unroll
        for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[wid][j*OFA_HG];
            const float pj[4] = {p0.x, p0.y, p0.z, p0.w};
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < 2*OFA_HG && !(lane & 1)) { s_m[warp][hg*OFA_HG + (lane >> 1)] = m_run; s_l[warp][hg*OFA_HG + (lane >> 1)] = l_run; }
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        float4 * dp = (float4 *) &s_o[warp][hg*OFA_HG + h][8*lane];
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
    __syncthreads();

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*OFA_G*OFA_D;
    for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += 2*OFA_NT) {
        const int h = idx / OFA_D, d = idx % OFA_D;
        float M = s_m[0][h];
#pragma unroll
        for (int w = 1; w < OFA_NW; ++w) M = fmaxf(M, s_m[w][h]);
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < OFA_NW; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][h] - M);
        po[idx] = acc;
        if (d == 0) {
            float L = 0.0f;
#pragma unroll
            for (int w = 0; w < OFA_NW; ++w) L += s_l[w][h]*exp2f(s_m[w][h] - M);
            p.part_ml[(size_t) blk*OFA_G + h] = make_float2(M, L);
        }
    }
}

// D1-ATT (12 Nov 2026), GGML_CUDA_OCCAMY_FA_ADDR=1 with PIPE=3: same kernel as ofa_decode_hs_kernel<false> but the K/V/mask
// row addresses are strength-reduced (pointer bumps) instead of a 64-bit row*nb1 multiply per row per step, which the
// 128-reg cap made ptxas rematerialise every iteration (~200 XMAD/IADD of ~700 loop instructions). Same loads, same FP ops,
// same order -> bitwise-identical output.
__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_ar_kernel(const ofa_params p) {
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int warp  = wid & (OFA_NW - 1);   // unit lane, same role as the flag-off warp index
    const int hg    = wid / OFA_NW;         // head group
    const int lane  = threadIdx.x & 31;

    __shared__ __align__(16) float s_p[2*OFA_NW][OFA_T*OFA_HG];
    __shared__ __align__(16) float s_a[2*OFA_NW][OFA_HG];
    __shared__ float  s_o[OFA_NW][OFA_G][OFA_D];
    __shared__ float  s_m[OFA_NW][OFA_G];
    __shared__ float  s_l[OFA_NW][OFA_G];

    float q[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(kvh*OFA_G + hg*OFA_HG + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    // after the reduce-scatter lane l holds value index l>>1 = token*4 + head (lanes l, l^1 duplicate)
    const int   jl = lane >> 3;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    const char * kp = Kb + (size_t)(u0 + warp)*OFA_T*nbk1;
    const char * vp = Vb + (size_t)(u0 + warp)*OFA_T*nbv1;
    const half * mp = Mb + (u0 + warp)*OFA_T + jl;
    const size_t kstep = (size_t) OFA_NW*OFA_T*nbk1;
    const size_t vstep = (size_t) OFA_NW*OFA_T*nbv1;
    for (int u = u0 + warp; u < u1; u += OFA_NW, kp += kstep, vp += vstep, mp += OFA_NW*OFA_T) {
        uint4 kr[OFA_T], vr[OFA_T];
        {
            const char * r = kp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbk1) kr[j] = __ldg((const uint4 *) r);
        }
        {
            const char * r = vp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbv1) vr[j] = __ldg((const uint4 *) r);
        }
        const float mk = __half2float(*mp);

        float s[OFA_T*OFA_HG];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*OFA_HG + h] = acc;
            }
        }
        // same lane pairing order as the flag-off kernel (xor 16, 8, 4, 2, 1)
        ofa_rs_step_x<8, 16>(s, lane);
        ofa_rs_step_x<4,  8>(s, lane);
        ofa_rs_step_x<2,  4>(s, lane);
        ofa_rs_step_x<1,  2>(s, lane);
        s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
        const float sc = s[0] + mk*1.4426950408889634f;
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
        if (lane < 2*OFA_HG && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
        __syncwarp();
        float a[OFA_HG];
        {
            const float4 a0 = *(const float4 *) &s_a[wid][0];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
        }
#pragma unroll
        for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[wid][j*OFA_HG];
            const float pj[4] = {p0.x, p0.y, p0.z, p0.w};
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < 2*OFA_HG && !(lane & 1)) { s_m[warp][hg*OFA_HG + (lane >> 1)] = m_run; s_l[warp][hg*OFA_HG + (lane >> 1)] = l_run; }
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        float4 * dp = (float4 *) &s_o[warp][hg*OFA_HG + h][8*lane];
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
    __syncthreads();

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*OFA_G*OFA_D;
    for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += 2*OFA_NT) {
        const int h = idx / OFA_D, d = idx % OFA_D;
        float M = s_m[0][h];
#pragma unroll
        for (int w = 1; w < OFA_NW; ++w) M = fmaxf(M, s_m[w][h]);
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < OFA_NW; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][h] - M);
        po[idx] = acc;
        if (d == 0) {
            float L = 0.0f;
#pragma unroll
            for (int w = 0; w < OFA_NW; ++w) L += s_l[w][h]*exp2f(s_m[w][h] - M);
            p.part_ml[(size_t) blk*OFA_G + h] = make_float2(M, L);
        }
    }
}

// Q8N (13 Nov 2026), GGML_CUDA_OCCAMY_KVQ8=1 (exact "1") with FA=1, PIPE=3, ADDR=1: the hs_ar kernel reading q8_0 K and/or V
// natively (no f16 conversion of the cache). KQ8/VQ8 select the K/V row format independently (F16/F16 stays on
// ofa_decode_hs_ar_kernel, untouched). Lane math: fattn-occamy-kvq8.cuh. K: the lane's 8-dim QK partial is computed on the
// int8 values and multiplied by its block scale d_k before the xor tree. V: p_j*d_v is formed once per (token, head).
// Same split/unit/softmax/combine order as hs_ar; not bitwise vs the generic q8 path (that one quantises Q to q8_1).
template <bool KQ8, bool VQ8>
__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_ar_q8_kernel(const ofa_params p) {
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int warp  = wid & (OFA_NW - 1);
    const int hg    = wid / OFA_NW;
    const int lane  = threadIdx.x & 31;

    __shared__ __align__(16) float s_p[2*OFA_NW][OFA_T*OFA_HG];
    __shared__ __align__(16) float s_a[2*OFA_NW][OFA_HG];
    __shared__ float  s_o[OFA_NW][OFA_G][OFA_D];
    __shared__ float  s_m[OFA_NW][OFA_G];
    __shared__ float  s_l[OFA_NW][OFA_G];

    float q[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(kvh*OFA_G + hg*OFA_HG + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    const int   jl = lane >> 3;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const ofq_lane g = ofq_lane_geom(lane);
    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + (KQ8 ? g.base : lane*16);
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + (VQ8 ? g.base : lane*16);
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    const char * kp = Kb + (size_t)(u0 + warp)*OFA_T*nbk1;
    const char * vp = Vb + (size_t)(u0 + warp)*OFA_T*nbv1;
    const half * mp = Mb + (u0 + warp)*OFA_T + jl;
    const size_t kstep = (size_t) OFA_NW*OFA_T*nbk1;
    const size_t vstep = (size_t) OFA_NW*OFA_T*nbv1;
    for (int u = u0 + warp; u < u1; u += OFA_NW, kp += kstep, vp += vstep, mp += OFA_NW*OFA_T) {
        uint4 kr[OFA_T], vr[OFA_T];      // f16: 8 halves; q8: x,y,z = words 0..2, w = f16 scale bits
        {
            const char * r = kp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbk1) {
                if (KQ8) {
                    kr[j].x = __ldg((const unsigned int *) r);
                    kr[j].y = __ldg((const unsigned int *) (r + 4));
                    kr[j].z = __ldg((const unsigned int *) (r + g.w2off));
                    kr[j].w = __ldg((const unsigned short *) (r + g.doff));
                } else {
                    kr[j] = __ldg((const uint4 *) r);
                }
            }
        }
        {
            const char * r = vp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbv1) {
                if (VQ8) {
                    vr[j].x = __ldg((const unsigned int *) r);
                    vr[j].y = __ldg((const unsigned int *) (r + 4));
                    vr[j].z = __ldg((const unsigned int *) (r + g.w2off));
                    vr[j].w = __ldg((const unsigned short *) (r + g.doff));
                } else {
                    vr[j] = __ldg((const uint4 *) r);
                }
            }
        }
        const float mk = __half2float(*mp);

        float s[OFA_T*OFA_HG];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            float dk = 1.0f;
            if (KQ8) {
                uint32_t lo, hi;
                ofq_realign(kr[j].x, kr[j].y, kr[j].z, g.sel, lo, hi);
                ofq_unpack8(lo, hi, k);
                dk = __half2float(__ushort_as_half((unsigned short) kr[j].w));
            } else {
                ofa_cvt8(kr[j], k);
            }
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*OFA_HG + h] = KQ8 ? acc*dk : acc;
            }
        }
        ofa_rs_step_x<8, 16>(s, lane);
        ofa_rs_step_x<4,  8>(s, lane);
        ofa_rs_step_x<2,  4>(s, lane);
        ofa_rs_step_x<1,  2>(s, lane);
        s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
        const float sc = s[0] + mk*1.4426950408889634f;
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
        if (lane < 2*OFA_HG && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
        __syncwarp();
        float a[OFA_HG];
        {
            const float4 a0 = *(const float4 *) &s_a[wid][0];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
        }
#pragma unroll
        for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            const float4 p0 = *(const float4 *) &s_p[wid][j*OFA_HG];
            float pj[4] = {p0.x, p0.y, p0.z, p0.w};
            if (VQ8) {
                uint32_t lo, hi;
                ofq_realign(vr[j].x, vr[j].y, vr[j].z, g.sel, lo, hi);
                ofq_unpack8(lo, hi, v);
                const float dv = __half2float(__ushort_as_half((unsigned short) vr[j].w));
#pragma unroll
                for (int h = 0; h < OFA_HG; ++h) pj[h] *= dv;
            } else {
                ofa_cvt8(vr[j], v);
            }
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < 2*OFA_HG && !(lane & 1)) { s_m[warp][hg*OFA_HG + (lane >> 1)] = m_run; s_l[warp][hg*OFA_HG + (lane >> 1)] = l_run; }
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        float4 * dp = (float4 *) &s_o[warp][hg*OFA_HG + h][8*lane];
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
    __syncthreads();

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
    float * po = p.part_o + (size_t) blk*OFA_G*OFA_D;
    for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += 2*OFA_NT) {
        const int h = idx / OFA_D, d = idx % OFA_D;
        float M = s_m[0][h];
#pragma unroll
        for (int w = 1; w < OFA_NW; ++w) M = fmaxf(M, s_m[w][h]);
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < OFA_NW; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][h] - M);
        po[idx] = acc;
        if (d == 0) {
            float L = 0.0f;
#pragma unroll
            for (int w = 0; w < OFA_NW; ++w) L += s_l[w][h]*exp2f(s_m[w][h] - M);
            p.part_ml[(size_t) blk*OFA_G + h] = make_float2(M, L);
        }
    }
}

// grid (nkvh*8 q-heads, ns), block 256 = one thread per dim
__global__ void __launch_bounds__(OFA_D) ofa_combine_kernel(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        char * __restrict__ dst, const int nsplit, const int nkvh, const size_t nbd1, const size_t nbd3) {
    const int qh = blockIdx.x, i3 = blockIdx.y, d = threadIdx.x;
    const int kvh = qh / OFA_G, h = qh % OFA_G;
    const int blk0 = (i3*nkvh + kvh)*nsplit;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, part_ml[(size_t)(blk0 + s)*OFA_G + h].x);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 ml = part_ml[(size_t)(blk0 + s)*OFA_G + h];
        const float  w  = exp2f(ml.x - M);
        L   += ml.y*w;
        acc += part_o[((size_t)(blk0 + s)*OFA_G + h)*OFA_D + d]*w;
    }
    float * out = (float *) (dst + i3*nbd3 + qh*nbd1);
    out[d] = L > 0.0f ? acc/L : 0.0f;
}

// compare kernel: res[0] = max|a-b| (as uint bits), res[1] = max|b|, res[2] = max rel (|a-b|/max(|b|,1e-2)), res[3] = nan count
__global__ void ofa_compare_kernel(const float * a, const float * b, const int n, unsigned int * res) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float x = a[i], y = b[i];
        if (!(x == x) || !(y == y)) { atomicAdd(&res[3], 1u); continue; }
        const float e = fabsf(x - y);
        atomicMax(&res[0], __float_as_uint(e));
        atomicMax(&res[1], __float_as_uint(fabsf(y)));
        atomicMax(&res[2], __float_as_uint(e/fmaxf(fabsf(y), 1e-2f)));
    }
}

int ggml_cuda_occamy_fa_mode() {
    static const int v = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA");
        int r = s ? atoi(s) : 0;
        if (r == 0) {
            const char * s2 = getenv("GGML_CUDA_NATIVE_FA_SM52");
            r = (s2 && atoi(s2) != 0) ? 1 : 0;
        }
        if (r) fprintf(stderr, "ggml_cuda: occamy FA decode kernel mode %d\n", r);
        return r;
    }();
    return v;
}

bool ggml_cuda_occamy_fa_applicable(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) return false;
    if (K->ne[0] != OFA_D || V->ne[0] != OFA_D || Q->ne[0] != OFA_D) return false;
    if (K->ne[2] != V->ne[2] || Q->ne[2] != OFA_G*K->ne[2]) return false;
    if (Q->ne[1] != 1) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1]) return false;
    if (K->ne[1] % OFA_T != 0) return false;
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1]) return false;
    if (Q->ne[3] % mask->ne[3] != 0) return false;
    if (sinks || max_bias != 0.0f || softcap != 0.0f) return false;
    if (Q->nb[0] != 4 || K->nb[0] != 2 || V->nb[0] != 2 || dst->nb[0] != 4) return false;
    if ((uintptr_t) Q->data % 16 || (uintptr_t) K->data % 16 || (uintptr_t) V->data % 16 || (uintptr_t) mask->data % 8) return false;
    for (int i = 1; i < 4; ++i) {
        if (Q->nb[i] % 16 || K->nb[i] % 16 || V->nb[i] % 16) return false;
    }
    return true;
}

// Q8N: GGML_CUDA_OCCAMY_KVQ8 must be exactly "1"; it only acts with GGML_CUDA_OCCAMY_FA_PIPE=3 and GGML_CUDA_OCCAMY_FA_ADDR!=0.
bool ggml_cuda_occamy_kvq8_enabled() {
    static const bool v = []() {
        const char * s  = getenv("GGML_CUDA_OCCAMY_KVQ8");
        const char * sp = getenv("GGML_CUDA_OCCAMY_FA_PIPE");
        const char * sa = getenv("GGML_CUDA_OCCAMY_FA_ADDR");
        const bool on = s && strcmp(s, "1") == 0;
        const bool ok = on && sp && atoi(sp) == 3 && sa && atoi(sa) != 0;
        if (on) fprintf(stderr, "ggml_cuda: occamy FA kvq8 %s\n", ok ? "on" : "requested but inactive (needs PIPE=3, ADDR=1)");
        return ok;
    }();
    return v;
}

static bool ofa_kv_ok(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F16) {
        if (t->nb[0] != 2 || (uintptr_t) t->data % 16) return false;
        for (int i = 1; i < 4; ++i) if (t->nb[i] % 16) return false;
        return true;
    }
    if (t->type == GGML_TYPE_Q8_0) {
        if (t->nb[0] != sizeof(block_q8_0) || (uintptr_t) t->data % 4) return false;
        for (int i = 1; i < 4; ++i) if (t->nb[i] % 4) return false;
        return true;
    }
    return false;
}

// same shape rules as ggml_cuda_occamy_fa_applicable; K/V each F16 or Q8_0 with at least one Q8_0
bool ggml_cuda_occamy_kvq8_applicable(const ggml_tensor * dst) {
    if (!ggml_cuda_occamy_kvq8_enabled()) return false;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (K->type != GGML_TYPE_Q8_0 && V->type != GGML_TYPE_Q8_0) return false;
    if (!ofa_kv_ok(K) || !ofa_kv_ok(V)) return false;
    if (K->ne[0] != OFA_D || V->ne[0] != OFA_D || Q->ne[0] != OFA_D) return false;
    if (K->ne[2] != V->ne[2] || Q->ne[2] != OFA_G*K->ne[2]) return false;
    if (Q->ne[1] != 1) return false;
    if (K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3] || K->ne[1] != V->ne[1]) return false;
    if (K->ne[1] % OFA_T != 0) return false;
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1]) return false;
    if (Q->ne[3] % mask->ne[3] != 0) return false;
    if (sinks || max_bias != 0.0f || softcap != 0.0f) return false;
    if (Q->nb[0] != 4 || dst->nb[0] != 4) return false;
    if ((uintptr_t) Q->data % 16 || (uintptr_t) mask->data % 8) return false;
    for (int i = 1; i < 4; ++i) if (Q->nb[i] % 16) return false;
    return true;
}

static void ofa_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, char * out, const bool kvq8 = false) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    static int resident = 0;
    static int waves = 1;
    static int pipe = 0;
    static int addr = 0;
    if (resident == 0) {
        int nb = 0;
        // occupancy of the flag-OFF kernel decides nsplit for every PIPE mode (same split -> same reduction);
        // the PIPE kernels must keep >= that occupancy, checked below.
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ofa_decode_kernel<0>, OFA_NT, 0));
        resident = std::max(1, nb) * ggml_cuda_info().devices[id].nsm;
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_WAVES");
        if (s) waves = std::max(1, atoi(s));
        const char * sp = getenv("GGML_CUDA_OCCAMY_FA_PIPE");
        pipe = sp ? atoi(sp) : 0;
        if (pipe < 0 || pipe > 4) pipe = 0;
        const char * sa = getenv("GGML_CUDA_OCCAMY_FA_ADDR");
        addr = (sa && atoi(sa) != 0 && pipe == 3) ? 1 : 0;
        if (pipe) {
            int nbp = 0;
            if (pipe <= 2) {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbp,
                    pipe == 1 ? ofa_decode_kernel<1> : ofa_decode_kernel<2>, OFA_NT, 0));
            } else {
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbp,
                    pipe == 3 ? ofa_decode_hs_kernel<false> : ofa_decode_hs_kernel<true>, 2*OFA_NT, 0));
                if (addr) {
                    int nba = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nba, ofa_decode_hs_ar_kernel, 2*OFA_NT, 0));
                    fprintf(stderr, "ggml_cuda: occamy FA addr mode 1 (blocks/SM %d)\n", nba);
                    if (nba < nb) addr = 0;
                    int nq = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nq, ofa_decode_hs_ar_q8_kernel<true, true>, 2*OFA_NT, 0));
                    if (ggml_cuda_occamy_kvq8_enabled()) fprintf(stderr, "ggml_cuda: occamy FA kvq8 blocks/SM %d\n", nq);
                }
            }
            fprintf(stderr, "ggml_cuda: occamy FA pipe mode %d (blocks/SM %d, flag-off %d)\n", pipe, nbp, nb);
        }
    }
    const int nkvh = (int) K->ne[2];
    const int ns   = (int) Q->ne[3];
    const int n_kv = (int) K->ne[1];
    int nsplit = (resident*waves + nkvh*ns - 1) / (nkvh*ns);
    nsplit = std::max(1, std::min(nsplit, n_kv / (OFA_T*OFA_NW*4)));  // >= 64 tokens per block

    const int nblk = nsplit*nkvh*ns;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) nblk*OFA_G*OFA_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) nblk*OFA_G);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ofa_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.n_kv = n_kv; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne1 = (int) mask->ne[1]; p.mask_ne3 = (int) mask->ne[3];
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];

    switch (pipe) {
        case 1:  ofa_decode_kernel<1><<<dim3(nsplit, nkvh, ns), OFA_NT, 0, stream>>>(p); break;
        case 2:  ofa_decode_kernel<2><<<dim3(nsplit, nkvh, ns), OFA_NT, 0, stream>>>(p); break;
        case 3:  if (kvq8) {
                     GGML_ASSERT(addr && "occamy kvq8 needs FA_ADDR=1");
                     const bool kq = K->type == GGML_TYPE_Q8_0, vq = V->type == GGML_TYPE_Q8_0;
                     if (kq && vq) ofa_decode_hs_ar_q8_kernel<true,  true ><<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p);
                     else if (kq)  ofa_decode_hs_ar_q8_kernel<true,  false><<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p);
                     else          ofa_decode_hs_ar_q8_kernel<false, true ><<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p);
                     break;
                 }
                 if (addr) { ofa_decode_hs_ar_kernel<<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p); break; }
                 ofa_decode_hs_kernel<false><<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p); break;
        case 4:  ofa_decode_hs_kernel<true ><<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p); break;
        default: GGML_ASSERT(!kvq8 && "occamy kvq8 needs FA_PIPE=3");
                 ofa_decode_kernel<0><<<dim3(nsplit, nkvh, ns), OFA_NT, 0, stream>>>(p); break;
    }
    CUDA_CHECK(cudaGetLastError());
    ofa_combine_kernel<<<dim3(nkvh*OFA_G, ns), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), out, nsplit, nkvh, dst->nb[1], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

// MRFA (17 Nov 2026): native multi-row F16 split-K decode, GGML_CUDA_OCCAMY_FA_MROW=1 (default off) for Q->ne[1] in 2..4.
// GGML_CUDA_OCCAMY_FA_MROW=2: run stock, then this kernel into a temp buffer, log max abs/rel diff, keep stock output.
// One CTA per (split, kv-head, stream); 2*R warps, warp w owns q-row r = w/2 and q-heads [4*(w&1), +4) of the kv-head
// (same 4-head x 8-dim lane layout, reduce-scatter and online softmax as ofa_decode_hs_ar_kernel). K/V rows of a tile of
// OFM_TT tokens are read from global once per CTA into shared memory and every row/head-group warp consumes them, so
// traffic stays at the 1-row level. Each warp walks all tokens of the split in order (no cross-warp combine): output
// is fp32-equivalent to the 1-row kernel, not bitwise. Mask row r is read at mask + r*nb[1]; dst row r at dst + r*nb[2]
// (dst is [D, n_head, n_q, ns]). Knobs: GGML_CUDA_OCCAMY_FA_MROW_PF (1 = register-prefetch the next tile, default 1),
// GGML_CUDA_OCCAMY_FA_MROW_NSPLIT (override split count; default see ofa_mrow_launch).
#define OFM_TT 16

template <int R, bool PF>
__global__ void __launch_bounds__(64*R, (R == 2 ? 4 : 2)) ofa_mrow_kernel(const ofa_params p) {
    constexpr int NT  = 64*R;
    constexpr int CPT = (OFM_TT*32 + NT - 1)/NT;   // 16-B chunks of K (and of V) per thread per tile
    constexpr int S   = OFA_G*R;                   // (row, head) slots per CTA
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int tid   = threadIdx.x;
    const int wid   = tid >> 5;
    const int r     = wid >> 1;          // q row
    const int hg    = wid & 1;           // head group
    const int lane  = tid & 31;

    __shared__ uint4 sK[OFM_TT*32];
    __shared__ uint4 sV[OFM_TT*32];
    __shared__ __align__(16) float s_p[2*R][OFA_T*OFA_HG];
    __shared__ __align__(16) float s_a[2*R][OFA_HG];

    float q[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t) r*p.nbq1 + (size_t)(kvh*OFA_G + hg*OFA_HG + h)*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[h][0] = a.x*p.scale_log2; q[h][1] = a.y*p.scale_log2; q[h][2] = a.z*p.scale_log2; q[h][3] = a.w*p.scale_log2;
        q[h][4] = b.x*p.scale_log2; q[h][5] = b.y*p.scale_log2; q[h][6] = b.z*p.scale_log2; q[h][7] = b.w*p.scale_log2;
    }
    float o[OFA_HG][8];
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }
    const int jl = lane >> 3;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2;
    const half * Mr = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3 + (size_t) r*p.nbm1);

    const int nunits = p.n_kv / OFA_T;
    const int t_beg = OFA_T*(int) (((long long) split      * nunits) / p.nsplit);
    const int t_end = OFA_T*(int) (((long long)(split + 1) * nunits) / p.nsplit);

    uint4 kr[CPT], vr[CPT];
    auto load_tile = [&](const int tb) {
        const int nc = min(OFM_TT, t_end - tb)*32;
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            const int c = tid + i*NT;
            if (c < nc) {
                const size_t row = (size_t) (tb + (c >> 5));
                kr[i] = __ldg((const uint4 *) (Kb + row*p.nbk1) + (c & 31));
                vr[i] = __ldg((const uint4 *) (Vb + row*p.nbv1) + (c & 31));
            }
        }
    };
    if constexpr (PF) {
        if (t_beg < t_end) load_tile(t_beg);
    }
    for (int tb = t_beg; tb < t_end; tb += OFM_TT) {
        const int nt = min(OFM_TT, t_end - tb);   // multiple of OFA_T
        if constexpr (!PF) load_tile(tb);
        __syncthreads();                           // previous tile consumed
#pragma unroll
        for (int i = 0; i < CPT; ++i) {
            const int c = tid + i*NT;
            if (c < nt*32 && (CPT*NT == OFM_TT*32 || c < OFM_TT*32)) { sK[c] = kr[i]; sV[c] = vr[i]; }
        }
        __syncthreads();
        if constexpr (PF) {
            if (tb + OFM_TT < t_end) load_tile(tb + OFM_TT);
        }
        for (int g = 0; g < nt; g += OFA_T) {
            const float mk = __half2float(Mr[tb + g + jl]);
            float s[OFA_T*OFA_HG];
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) {
                float k[8];
                ofa_cvt8(sK[(g + j)*32 + lane], k);
#pragma unroll
                for (int h = 0; h < OFA_HG; ++h) {
                    float acc = q[h][0]*k[0];
#pragma unroll
                    for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                    s[j*OFA_HG + h] = acc;
                }
            }
            ofa_rs_step_x<8, 16>(s, lane);
            ofa_rs_step_x<4,  8>(s, lane);
            ofa_rs_step_x<2,  4>(s, lane);
            ofa_rs_step_x<1,  2>(s, lane);
            s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
            const float sc = s[0] + mk*1.4426950408889634f;
            float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
            const float m_new = fmaxf(m_run, gm);
            const float alpha = exp2f(m_run - m_new);
            const float pr    = exp2f(sc - m_new);
            m_run = m_new;
            l_run = l_run*alpha + pr;
            if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
            if (lane < 2*OFA_HG && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
            __syncwarp();
            float a[OFA_HG];
            {
                const float4 a0 = *(const float4 *) &s_a[wid][0];
                a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
            }
#pragma unroll
            for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
            }
#pragma unroll
            for (int j = 0; j < OFA_T; ++j) {
                float v[8];
                ofa_cvt8(sV[(g + j)*32 + lane], v);
                const float4 p0 = *(const float4 *) &s_p[wid][j*OFA_HG];
                const float pj[4] = {p0.x, p0.y, p0.z, p0.w};
#pragma unroll
                for (int h = 0; h < OFA_HG; ++h) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
                }
            }
            __syncwarp();
        }
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    const int blk  = (i3*p.nkvh + kvh)*p.nsplit + split;
    const int slot = r*OFA_G + hg*OFA_HG;
    if (lane < 2*OFA_HG && !(lane & 1)) p.part_ml[(size_t) blk*S + slot + (lane >> 1)] = make_float2(m_run, l_run);
#pragma unroll
    for (int h = 0; h < OFA_HG; ++h) {
        float4 * dp = (float4 *) (p.part_o + ((size_t) blk*S + slot + h)*OFA_D + 8*lane);
        dp[0] = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
        dp[1] = make_float4(o[h][4], o[h][5], o[h][6], o[h][7]);
    }
}

// grid (nkvh*8 q-heads, n_q, ns), block 256
__global__ void __launch_bounds__(OFA_D) ofa_mrow_combine_kernel(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
        char * __restrict__ dst, const int nsplit, const int nkvh, const int nq, const size_t nbd1, const size_t nbd2, const size_t nbd3) {
    const int qh = blockIdx.x, r = blockIdx.y, i3 = blockIdx.z, d = threadIdx.x;
    const int kvh = qh / OFA_G, h = qh % OFA_G;
    const int S = OFA_G*nq, slot = r*OFA_G + h;
    const int blk0 = (i3*nkvh + kvh)*nsplit;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, part_ml[(size_t)(blk0 + s)*S + slot].x);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float2 ml = part_ml[(size_t)(blk0 + s)*S + slot];
        const float  w  = exp2f(ml.x - M);
        L   += ml.y*w;
        acc += part_o[((size_t)(blk0 + s)*S + slot)*OFA_D + d]*w;
    }
    float * out = (float *) (dst + i3*nbd3 + (size_t) r*nbd2 + (size_t) qh*nbd1);
    out[d] = L > 0.0f ? acc/L : 0.0f;
}

int ggml_cuda_occamy_fa_mrow_mode() {
    static const int v = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_MROW");
        const int r = s ? atoi(s) : 0;
        if (r) fprintf(stderr, "ggml_cuda: occamy FA mrow mode %d\n", r);
        return r;
    }();
    return v;
}

bool ggml_cuda_occamy_fa_mrow_applicable(const ggml_tensor * dst) {
    if (ggml_cuda_occamy_fa_mrow_mode() == 0) return false;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    if (Q->ne[1] < 2 || Q->ne[1] > 4) return false;
    if (!mask || mask->ne[1] < Q->ne[1]) return false;
    // all other conditions are the 1-row ones, checked on a 1-row view (data/strides unchanged)
    ggml_tensor d1 = *dst, q1 = *Q;
    q1.ne[1] = 1;
    d1.src[0] = &q1;
    return ggml_cuda_occamy_fa_applicable(&d1);
}

static std::atomic<long long> ofm_calls[5];

static void ofa_mrow_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, char * out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();
    const int nq = (int) Q->ne[1];

    static int resident = 0, waves = 1, pf = 1, nsplit_env = 0, res_r2 = 0;
    if (resident == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ofa_decode_kernel<0>, OFA_NT, 0));
        resident = std::max(1, nb) * ggml_cuda_info().devices[id].nsm;
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_WAVES");
        if (s) waves = std::max(1, atoi(s));
        const char * sp = getenv("GGML_CUDA_OCCAMY_FA_MROW_PF");
        if (sp) pf = atoi(sp) != 0;
        const char * sn = getenv("GGML_CUDA_OCCAMY_FA_MROW_NSPLIT");
        if (sn) nsplit_env = std::max(0, atoi(sn));
        int o2 = 0, o3 = 0, o4 = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2, pf ? ofa_mrow_kernel<2, true> : ofa_mrow_kernel<2, false>, 128, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o3, pf ? ofa_mrow_kernel<3, true> : ofa_mrow_kernel<3, false>, 192, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o4, pf ? ofa_mrow_kernel<4, true> : ofa_mrow_kernel<4, false>, 256, 0));
        res_r2 = std::max(1, o2) * ggml_cuda_info().devices[id].nsm;
        fprintf(stderr, "ggml_cuda: occamy FA mrow pf %d nsplit_env %d, 1-row resident %d, mrow blocks/SM R2 %d R3 %d R4 %d (nsm %d)\n",
            pf, nsplit_env, resident, o2, o3, o4, ggml_cuda_info().devices[id].nsm);
    }
    const int nkvh = (int) K->ne[2];
    const int ns   = (int) Q->ne[3];
    const int n_kv = (int) K->ne[1];
    // default: R=2 -> 4 waves of mrow R2 CTAs (80K sweep: 24 splits 1.62 ms, 64 1.54, 192 1.24); R=3/4 -> 1-row split
    // count (fewer splits measured best there: R4 24 splits 1.79 ms vs 64/192 1.93).
    int nsplit = nsplit_env > 0 ? nsplit_env
               : nq == 2 ? (4*res_r2 + nkvh*ns - 1) / (nkvh*ns)
                         : (resident*waves + nkvh*ns - 1) / (nkvh*ns);
    nsplit = std::max(1, std::min(nsplit, n_kv / (OFA_T*OFA_NW*4)));

    const long long nc = ++ofm_calls[nq];
    if (nc == 1 || nc == 10 || nc == 100 || nc == 1000 || nc % 10000 == 0) {
        fprintf(stderr, "ggml_cuda: occamy FA mrow dispatch n_q=%d calls=%lld n_kv=%d nkvh=%d ns=%d nsplit=%d\n", nq, nc, n_kv, nkvh, ns, nsplit);
    }

    const int nblk = nsplit*nkvh*ns;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) nblk*OFA_G*nq*OFA_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) nblk*OFA_G*nq);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ofa_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.n_kv = n_kv; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne1 = (int) mask->ne[1]; p.mask_ne3 = (int) mask->ne[3];
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];

    const dim3 grid(nsplit, nkvh, ns);
    switch (nq) {
        case 2: if (pf) ofa_mrow_kernel<2, true><<<grid, 128, 0, stream>>>(p); else ofa_mrow_kernel<2, false><<<grid, 128, 0, stream>>>(p); break;
        case 3: if (pf) ofa_mrow_kernel<3, true><<<grid, 192, 0, stream>>>(p); else ofa_mrow_kernel<3, false><<<grid, 192, 0, stream>>>(p); break;
        case 4: if (pf) ofa_mrow_kernel<4, true><<<grid, 256, 0, stream>>>(p); else ofa_mrow_kernel<4, false><<<grid, 256, 0, stream>>>(p); break;
        default: GGML_ABORT("occamy FA mrow: n_q must be 2..4");
    }
    CUDA_CHECK(cudaGetLastError());
    ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, nq, ns), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), out, nsplit, nkvh, nq,
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

// FA2 (17 Nov 2026), GGML_CUDA_OCCAMY_FA_MROW=3: n_q = 2 split-K decode built on the 1-row hs_ar kernel (n_q 3..4 keep the
// MROW=1 kernel). One CTA per (split, kv-head, stream), NU unit-lane warps x 4 head-pair groups. Warp (u, g) walks units
// u0+u, u0+u+NU, ... of 4 tokens like hs_ar, and owns 4 (row, head) slots: rows 0,1 x q-heads {2g, 2g+1}. Each K/V row is
// loaded and converted f16 -> f32 once per warp and used for both query rows (dual online-softmax state per slot); the 4
// head-pair groups re-read the same rows (L1/tex hits, as the 2 head groups of hs_ar). Per-lane register budget = hs_ar
// (4 slots x 8 dims of q and o). After the reduce-scatter lane l holds (token l>>3, slot (l>>1)&3), slot = row*2 + head,
// so lane l reads mask row (l>>2)&1. Warps of a CTA are combined in shared memory (one row per pass) into the MROW
// partial layout [blk][row*8 + head][D] and ofa_mrow_combine_kernel finishes. Knobs: GGML_CUDA_OCCAMY_FA2_NU (2 or 4,
// default 4), GGML_CUDA_OCCAMY_FA2_V (1 default, 2 = 8-slot ofa_fa3_kernel, slower: 0.91 vs 0.89 ms), GGML_CUDA_OCCAMY_FA2_WAVES (default 1), GGML_CUDA_OCCAMY_FA_MROW_NSPLIT (override split count).
#define OF2_NG 4

template <int NU>
__global__ void __launch_bounds__(32*NU*OF2_NG, (NU == 2 ? 2 : 1)) ofa_fa2_kernel(const ofa_params p) {
    constexpr int NT = 32*NU*OF2_NG;
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int ul    = wid % NU;      // unit lane
    const int g     = wid / NU;      // head pair: q-heads 2g, 2g+1
    const int lane  = threadIdx.x & 31;

    __shared__ __align__(16) float s_p[NU*OF2_NG][OFA_T*4];
    __shared__ __align__(16) float s_a[NU*OF2_NG][4];
    __shared__ __align__(16) float s_o[NU][OFA_G][OFA_D];
    __shared__ float s_m[NU][2*OFA_G];
    __shared__ float s_l[NU][2*OFA_G];

    // slot k: row k>>1, q-head 2g + (k&1)
    float q[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(k >> 1)*p.nbq1 + (size_t)(kvh*OFA_G + 2*g + (k & 1))*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[k][0] = a.x*p.scale_log2; q[k][1] = a.y*p.scale_log2; q[k][2] = a.z*p.scale_log2; q[k][3] = a.w*p.scale_log2;
        q[k][4] = b.x*p.scale_log2; q[k][5] = b.y*p.scale_log2; q[k][6] = b.z*p.scale_log2; q[k][7] = b.w*p.scale_log2;
    }
    float o[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[k][i] = 0.0f;
    }
    const int jl = lane >> 3;            // token in the unit whose score this lane owns
    const int rl = (lane >> 2) & 1;      // q row of that score
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3 + (size_t) rl*p.nbm1);

    const int nunits = p.n_kv / OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    const char * kp = Kb + (size_t)(u0 + ul)*OFA_T*nbk1;
    const char * vp = Vb + (size_t)(u0 + ul)*OFA_T*nbv1;
    const half * mp = Mb + (u0 + ul)*OFA_T + jl;
    const size_t kstep = (size_t) NU*OFA_T*nbk1;
    const size_t vstep = (size_t) NU*OFA_T*nbv1;
    for (int u = u0 + ul; u < u1; u += NU, kp += kstep, vp += vstep, mp += NU*OFA_T) {
        uint4 kr[OFA_T], vr[OFA_T];
        {
            const char * r = kp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbk1) kr[j] = __ldg((const uint4 *) r);
        }
        {
            const char * r = vp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbv1) vr[j] = __ldg((const uint4 *) r);
        }
        const float mk = __half2float(*mp);

        float s[OFA_T*4];
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*4 + h] = acc;
            }
        }
        ofa_rs_step_x<8, 16>(s, lane);
        ofa_rs_step_x<4,  8>(s, lane);
        ofa_rs_step_x<2,  4>(s, lane);
        ofa_rs_step_x<1,  2>(s, lane);
        s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
        const float sc = s[0] + mk*1.4426950408889634f;
        float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 8, 32));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 16, 32));
        const float m_new = fmaxf(m_run, gm);
        const float alpha = exp2f(m_run - m_new);
        const float pr    = exp2f(sc - m_new);
        m_run = m_new;
        l_run = l_run*alpha + pr;
        if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
        if (lane < 8 && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
        __syncwarp();
        float a[4];
        {
            const float4 a0 = *(const float4 *) &s_a[wid][0];
            a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
        }
#pragma unroll
        for (int h = 0; h < 4; ++h) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
        }
#pragma unroll
        for (int j = 0; j < OFA_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[wid][j*4];
            const float pj[4] = {p0.x, p0.y, p0.z, p0.w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 8, 32);
    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < 8 && !(lane & 1)) {
        const int k = lane >> 1;
        const int slot = (k >> 1)*OFA_G + 2*g + (k & 1);
        s_m[ul][slot] = m_run; s_l[ul][slot] = l_run;
    }

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r) __syncthreads();   // pass 0 reads of s_o done
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            float4 * dp = (float4 *) &s_o[ul][2*g + hh][8*lane];
            dp[0] = make_float4(o[2*r + hh][0], o[2*r + hh][1], o[2*r + hh][2], o[2*r + hh][3]);
            dp[1] = make_float4(o[2*r + hh][4], o[2*r + hh][5], o[2*r + hh][6], o[2*r + hh][7]);
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += NT) {
            const int h = idx / OFA_D, d = idx % OFA_D;
            const int slot = r*OFA_G + h;
            float M = s_m[0][slot];
#pragma unroll
            for (int w = 1; w < NU; ++w) M = fmaxf(M, s_m[w][slot]);
            float acc = 0.0f;
#pragma unroll
            for (int w = 0; w < NU; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][slot] - M);
            p.part_o[((size_t) blk*2*OFA_G + slot)*OFA_D + d] = acc;
            if (d == 0) {
                float L = 0.0f;
#pragma unroll
                for (int w = 0; w < NU; ++w) L += s_l[w][slot]*exp2f(s_m[w][slot] - M);
                p.part_ml[(size_t) blk*2*OFA_G + slot] = make_float2(M, L);
            }
        }
    }
}

// FA2 v2 (GGML_CUDA_OCCAMY_FA2_V=2): 8 slots per warp (rows 0,1 x q-heads 4g..4g+3, g = 0,1), units of 2 tokens. Each
// K/V row is converted by 2 warps (as in the 1-row kernel) instead of 4, and the reduce-scatter covers 16 values per
// 2 tokens x 8 slots. The o rescale is skipped when no slot's running max moved (warp vote). After the reduce-scatter lane
// l holds (token l>>4, slot (l>>1)&7), slot = row*4 + head-in-group, so lane l reads mask row (l>>3)&1.
#define OF3_T  2
#define OF3_NG 2

template <int NU, int MINB>
__global__ void __launch_bounds__(32*NU*OF3_NG, MINB) ofa_fa3_kernel(const ofa_params p) {
    constexpr int NT = 32*NU*OF3_NG;
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int ul    = wid % NU;
    const int g     = wid / NU;      // q-heads 4g..4g+3
    const int lane  = threadIdx.x & 31;

    __shared__ __align__(16) float s_p[NU*OF3_NG][OF3_T*8];
    __shared__ __align__(16) float s_a[NU*OF3_NG][8];
    __shared__ __align__(16) float s_o[NU][OFA_G][OFA_D];
    __shared__ float s_m[NU][2*OFA_G];
    __shared__ float s_l[NU][2*OFA_G];

    float q[8][8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t)(k >> 2)*p.nbq1 + (size_t)(kvh*OFA_G + 4*g + (k & 3))*p.nbq2) + 2*lane;
        const float4 a = qp[0], b = qp[1];
        q[k][0] = a.x*p.scale_log2; q[k][1] = a.y*p.scale_log2; q[k][2] = a.z*p.scale_log2; q[k][3] = a.w*p.scale_log2;
        q[k][4] = b.x*p.scale_log2; q[k][5] = b.y*p.scale_log2; q[k][6] = b.z*p.scale_log2; q[k][7] = b.w*p.scale_log2;
    }
    float o[8][8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
#pragma unroll
        for (int i = 0; i < 8; ++i) o[k][i] = 0.0f;
    }
    const int jl = lane >> 4;
    const int rl = (lane >> 3) & 1;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3 + (size_t) rl*p.nbm1);

    const int nunits = p.n_kv / OF3_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    const char * kp = Kb + (size_t)(u0 + ul)*OF3_T*nbk1;
    const char * vp = Vb + (size_t)(u0 + ul)*OF3_T*nbv1;
    const half * mp = Mb + (u0 + ul)*OF3_T + jl;
    const size_t kstep = (size_t) NU*OF3_T*nbk1;
    const size_t vstep = (size_t) NU*OF3_T*nbv1;
    // register double buffer: the next unit's K/V/mask loads are issued before this unit's math (8 warps/SM only)
    uint4 kn[OF3_T], vn[OF3_T];
    half  mn = __float2half(0.0f);
    if (u0 + ul < u1) {
#pragma unroll
        for (int j = 0; j < OF3_T; ++j) kn[j] = __ldg((const uint4 *) (kp + j*nbk1));
#pragma unroll
        for (int j = 0; j < OF3_T; ++j) vn[j] = __ldg((const uint4 *) (vp + j*nbv1));
        mn = *mp;
    }
    for (int u = u0 + ul; u < u1; u += NU, kp += kstep, vp += vstep, mp += NU*OF3_T) {
        uint4 kr[OF3_T], vr[OF3_T];
#pragma unroll
        for (int j = 0; j < OF3_T; ++j) { kr[j] = kn[j]; vr[j] = vn[j]; }
        const float mk = __half2float(mn);
        if (u + NU < u1) {
#pragma unroll
            for (int j = 0; j < OF3_T; ++j) kn[j] = __ldg((const uint4 *) (kp + kstep + j*nbk1));
#pragma unroll
            for (int j = 0; j < OF3_T; ++j) vn[j] = __ldg((const uint4 *) (vp + vstep + j*nbv1));
            mn = mp[NU*OF3_T];
        }

        float s[OF3_T*8];
#pragma unroll
        for (int j = 0; j < OF3_T; ++j) {
            float k[8];
            ofa_cvt8(kr[j], k);
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                float acc = q[h][0]*k[0];
#pragma unroll
                for (int i = 1; i < 8; ++i) acc = fmaf(q[h][i], k[i], acc);
                s[j*8 + h] = acc;
            }
        }
        ofa_rs_step_x<8, 16>(s, lane);
        ofa_rs_step_x<4,  8>(s, lane);
        ofa_rs_step_x<2,  4>(s, lane);
        ofa_rs_step_x<1,  2>(s, lane);
        s[0] = s[0] + __shfl_xor_sync(0xFFFFFFFF, s[0], 1, 32);
        const float sc = s[0] + mk*1.4426950408889634f;
        const float gm = fmaxf(sc, __shfl_xor_sync(0xFFFFFFFF, sc, 16, 32));
        const bool  up = gm > m_run;
        const float m_new = up ? gm : m_run;
        const float pr    = exp2f(sc - m_new);
        if (!(lane & 1)) s_p[wid][lane >> 1] = pr;
        if (__any_sync(0xFFFFFFFF, up)) {
            const float alpha = exp2f(m_run - m_new);
            l_run *= alpha;
            if (lane < 16 && !(lane & 1)) s_a[wid][lane >> 1] = alpha;
            __syncwarp();
            const float4 a0 = *(const float4 *) &s_a[wid][0];
            const float4 a1 = *(const float4 *) &s_a[wid][4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int h = 0; h < 8; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] *= a[h];
            }
        } else {
            __syncwarp();
        }
        m_run = m_new;
        l_run += pr;
#pragma unroll
        for (int j = 0; j < OF3_T; ++j) {
            float v[8];
            ofa_cvt8(vr[j], v);
            const float4 p0 = *(const float4 *) &s_p[wid][j*8];
            const float4 p1 = *(const float4 *) &s_p[wid][j*8 + 4];
            const float pj[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int h = 0; h < 8; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = fmaf(pj[h], v[i], o[h][i]);
            }
        }
        __syncwarp();
    }

    l_run += __shfl_xor_sync(0xFFFFFFFF, l_run, 16, 32);
    if (lane < 16 && !(lane & 1)) {
        const int k = lane >> 1;
        const int slot = (k >> 2)*OFA_G + 4*g + (k & 3);
        s_m[ul][slot] = m_run; s_l[ul][slot] = l_run;
    }

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r) __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            float4 * dp = (float4 *) &s_o[ul][4*g + hh][8*lane];
            dp[0] = make_float4(o[4*r + hh][0], o[4*r + hh][1], o[4*r + hh][2], o[4*r + hh][3]);
            dp[1] = make_float4(o[4*r + hh][4], o[4*r + hh][5], o[4*r + hh][6], o[4*r + hh][7]);
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < OFA_G*OFA_D; idx += NT) {
            const int h = idx / OFA_D, d = idx % OFA_D;
            const int slot = r*OFA_G + h;
            float M = s_m[0][slot];
#pragma unroll
            for (int w = 1; w < NU; ++w) M = fmaxf(M, s_m[w][slot]);
            float acc = 0.0f;
#pragma unroll
            for (int w = 0; w < NU; ++w) acc += s_o[w][h][d]*exp2f(s_m[w][slot] - M);
            p.part_o[((size_t) blk*2*OFA_G + slot)*OFA_D + d] = acc;
            if (d == 0) {
                float L = 0.0f;
#pragma unroll
                for (int w = 0; w < NU; ++w) L += s_l[w][slot]*exp2f(s_m[w][slot] - M);
                p.part_ml[(size_t) blk*2*OFA_G + slot] = make_float2(M, L);
            }
        }
    }
}

#define OF3_K4 ofa_fa3_kernel<4, 1>
#define OF3_K2 ofa_fa3_kernel<2, OF3_MINB2>
#ifndef OF3_MINB2
#define OF3_MINB2 2
#endif

static void ofa_fa2_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, char * out) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();
    GGML_ASSERT(Q->ne[1] == 2);

    static int resident = 0, nu = 4, waves = 1, nsplit_env = 0, ver = 1;
    if (resident == 0) {
        const char * su = getenv("GGML_CUDA_OCCAMY_FA2_NU");
        if (su) nu = atoi(su) == 2 ? 2 : 4;
        const char * sv = getenv("GGML_CUDA_OCCAMY_FA2_V");
        if (sv) ver = atoi(sv) == 2 ? 2 : 1;
        const char * sw = getenv("GGML_CUDA_OCCAMY_FA2_WAVES");
        if (sw) waves = std::max(1, atoi(sw));
        const char * sn = getenv("GGML_CUDA_OCCAMY_FA_MROW_NSPLIT");
        if (sn) nsplit_env = std::max(0, atoi(sn));
        int nb = 0;
        if (ver == 2) {
            if (nu == 4) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, OF3_K4, 32*4*OF3_NG, 0));
            else         CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, OF3_K2, 32*2*OF3_NG, 0));
        } else {
            if (nu == 4) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ofa_fa2_kernel<4>, 32*4*OF2_NG, 0));
            else         CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, ofa_fa2_kernel<2>, 32*2*OF2_NG, 0));
        }
        resident = std::max(1, nb) * ggml_cuda_info().devices[id].nsm;
        fprintf(stderr, "ggml_cuda: occamy FA2 (n_q=2) v%d nu %d waves %d nsplit_env %d blocks/SM %d resident %d\n",
            ver, nu, waves, nsplit_env, nb, resident);
    }
    const int nkvh = (int) K->ne[2];
    const int ns   = (int) Q->ne[3];
    const int n_kv = (int) K->ne[1];
    int nsplit = nsplit_env > 0 ? nsplit_env : (resident*waves + nkvh*ns - 1) / (nkvh*ns);
    nsplit = std::max(1, std::min(nsplit, n_kv / ((ver == 2 ? OF3_T : OFA_T)*nu*4)));

    const long long nc = ++ofm_calls[0];
    if (nc == 1 || nc == 10 || nc == 100 || nc == 1000 || nc % 10000 == 0) {
        fprintf(stderr, "ggml_cuda: occamy FA2 dispatch calls=%lld n_kv=%d nkvh=%d ns=%d nsplit=%d\n", nc, n_kv, nkvh, ns, nsplit);
    }

    const int nblk = nsplit*nkvh*ns;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) nblk*2*OFA_G*OFA_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) nblk*2*OFA_G);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ofa_params p;
    p.Q = (const char *) Q->data; p.K = (const char *) K->data; p.V = (const char *) V->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.n_kv = n_kv; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne1 = (int) mask->ne[1]; p.mask_ne3 = (int) mask->ne[3];
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2]; p.nbq3 = Q->nb[3];
    p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2]; p.nbk3 = K->nb[3];
    p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2]; p.nbv3 = V->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];

    const dim3 grid(nsplit, nkvh, ns);
    if (ver == 2) {
        if (nu == 4) OF3_K4<<<grid, 32*4*OF3_NG, 0, stream>>>(p);
        else         OF3_K2<<<grid, 32*2*OF3_NG, 0, stream>>>(p);
    } else {
        if (nu == 4) ofa_fa2_kernel<4><<<grid, 32*4*OF2_NG, 0, stream>>>(p);
        else         ofa_fa2_kernel<2><<<grid, 32*2*OF2_NG, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());
    ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, 2, ns), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), out, nsplit, nkvh, 2,
        dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}

#include "fattn-occamy-sparse.cuh"

bool ggml_cuda_occamy_fa_mrow_is_fa2(const ggml_tensor * dst) {
    return ggml_cuda_occamy_fa_mrow_mode() == 3 && dst->src[0]->ne[1] == 2;
}

void ggml_cuda_occamy_fa_mrow(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_occamy_fa_mrow_is_fa2(dst)) {
        ofa_fa2_launch(ctx, dst, (char *) dst->data);
        return;
    }
    ofa_mrow_launch(ctx, dst, (char *) dst->data);
}

// MROW=2: stock result already in dst; run mrow into a temp buffer and log the diff for every call (first 400, then 1/200).
void ggml_cuda_occamy_fa_mrow_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static int ncalls = 0;
    ++ncalls;
    if (!(ncalls <= 400 || ncalls % 200 == 0)) return;
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int n = (int) ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);
    ofa_mrow_launch(ctx, dst, (char *) tmp.get());
    ggml_cuda_pool_alloc<unsigned int> res(ctx.pool(), 4);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(res.get(), 0, 4*sizeof(unsigned int), stream));
    ofa_compare_kernel<<<16, 256, 0, stream>>>(tmp.get(), (const float *) dst->data, n, res.get());
    unsigned int h[4];
    CUDA_CHECK(cudaMemcpyAsync(h, res.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float f[3];
    memcpy(f, h, sizeof(f));
    fprintf(stderr, "occamy_fa_mrow_cmp call=%d name=%s n_q=%lld n_kv=%lld ns=%lld max_abs=%.3e max_ref=%.3e max_rel=%.3e nan=%u\n",
        ncalls, dst->name, (long long) dst->src[0]->ne[1], (long long) dst->src[1]->ne[1], (long long) dst->src[0]->ne[3], f[0], f[1], f[2], h[3]);
}

// MV3: 2-row verify as two 1-row launches (GGML_CUDA_OCCAMY_FA_2CALL=1, default off).
// Builds 1-row views of Q/mask/dst per row and reuses the 1-row applicable() checks.
bool ggml_cuda_occamy_fa_2call_enabled() {
    static const bool v = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_2CALL");
        return s && strcmp(s, "1") == 0;
    }();
    return v;
}

static bool ofa_2call_views(const ggml_tensor * dst, ggml_tensor * d1, ggml_tensor * q1, ggml_tensor * m1, int r) {
    *d1 = *dst; *q1 = *dst->src[0]; *m1 = *dst->src[3];
    q1->ne[1] = 1; m1->ne[1] = 1; d1->ne[2] = 1;   // dst is [D, n_head, n_q, ns]: rows are ne[2]/nb[2]
    q1->data = (char *) dst->src[0]->data + (size_t) r*dst->src[0]->nb[1];
    m1->data = (char *) dst->src[3]->data + (size_t) r*dst->src[3]->nb[1];
    d1->data = (char *) dst->data + (size_t) r*dst->nb[2];
    d1->src[0] = q1; d1->src[3] = m1;
    return true;
}

bool ggml_cuda_occamy_fa_2call_applicable(const ggml_tensor * dst) {
    if (!ggml_cuda_occamy_fa_2call_enabled()) return false;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    if (Q->ne[1] != 2 || !mask || mask->ne[1] < 2) return false;
    ggml_tensor d1, q1, m1;
    for (int r = 0; r < 2; ++r) {
        ofa_2call_views(dst, &d1, &q1, &m1, r);
        if (!ggml_cuda_occamy_fa_applicable(&d1)) return false;  // checks data%16 (Q), mask%8
    }
    return true;
}

void ggml_cuda_occamy_fa_2call(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor d1, q1, m1;
    for (int r = 0; r < 2; ++r) {
        ofa_2call_views(dst, &d1, &q1, &m1, r);
        ofa_launch(ctx, &d1, (char *) d1.data);
    }
}

void ggml_cuda_occamy_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ofa_launch(ctx, dst, (char *) dst->data);
}

void ggml_cuda_occamy_fa_kvq8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ofa_launch(ctx, dst, (char *) dst->data, true);
}

// mode 2: our result into a temp buffer, stock result already in dst; compare and log.
void ggml_cuda_occamy_fa_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool kvq8) {
    static int ncalls = 0;
    ++ncalls;
    const bool log_it = ncalls <= 40 || ncalls % 200 == 0;
    if (!log_it) return;
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int n = (int) ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);
    // dst is [D, nh, 1, ns] contiguous: our launch writes with dst strides into tmp
    ofa_launch(ctx, dst, (char *) tmp.get(), kvq8);
    ggml_cuda_pool_alloc<unsigned int> res(ctx.pool(), 4);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(res.get(), 0, 4*sizeof(unsigned int), stream));
    ofa_compare_kernel<<<16, 256, 0, stream>>>(tmp.get(), (const float *) dst->data, n, res.get());
    unsigned int h[4];
    CUDA_CHECK(cudaMemcpyAsync(h, res.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float f[3];
    memcpy(f, h, sizeof(f));
    fprintf(stderr, "occamy_fa_cmp%s call=%d n_kv=%lld ns=%lld max_abs=%.3e max_ref=%.3e max_rel=%.3e nan=%u\n",
        kvq8 ? "_kvq8" : "", ncalls, (long long) dst->src[1]->ne[1], (long long) dst->src[0]->ne[3], f[0], f[1], f[2], h[3]);
}
