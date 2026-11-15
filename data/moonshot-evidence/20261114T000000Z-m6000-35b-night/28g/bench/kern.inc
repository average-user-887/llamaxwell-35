namespace occamy {

constexpr int H = 2048, NE = 256, TOPK = 8, FF = 512, FFS = 512;
constexpr int QB = 18;
constexpr int ROW_GU = H / 32 * QB;    // 1152 B
constexpr int ROW_D  = FF / 32 * QB;   // 288 B
constexpr int NT = 256, NW = NT / 32;
constexpr int CH = 68;
constexpr int NQ = 4, QK = H / NQ;
constexpr int NTA = NQ * (NE + 1), NTB = FFS + TOPK * FF;
constexpr int STW = 648;
constexpr int SMEM_FLOATS  = (TOPK * FF + FFS) / 64 * CH + (TOPK * FF + FFS) / 32 * 2 / 2 * 2 + 64;
constexpr int SMEM2_FLOATS = SMEM_FLOATS + NW * STW;
constexpr unsigned SPIN_LIMIT = 1u << 26;   // ~seconds of spinning -> a non-co-resident grid traps instead of hanging

struct Layer {
    const float * gate_inp;
    const float * gate_inp_sh;
    const uint8_t * gate_exps, * up_exps, * down_exps;
    const uint8_t * gate_sh, * up_sh, * down_sh;
};
struct Scratch {
    float logits[NE];
    float h_sh[FFS];
    float gsh;
    float h_exp[TOPK * FF];
    float lpart[NQ][NE], gpart[NQ];
    unsigned bar_count, bar_gen;
};
struct Params {
    Layer L;
    const float * h;
    const float * norm;
    float * out;
    Scratch * s;
    float eps;
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float wsum(float v) {
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
            while (*genp == gen) {
                if (++k > SPIN_LIMIT) { __trap(); }
            }
        }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ float nib(unsigned v, int k) {
    return __int_as_float(__byte_perm(v, 0x3F000000u, 0x7044u | (k << 8)));
}
struct Q9 { unsigned w[9]; };
__device__ __forceinline__ Q9 lds9(const unsigned * p) { Q9 q;
#pragma unroll
    for (int i = 0; i < 9; ++i) q.w[i] = p[i];
    return q; }
// MOE1 (GGML_CUDA_MOE_LOP3=1, DX bit 16): nibble unpack (v & 0x0F0F0F0F) | 0x80808080 as one LOP3 instead of
// AND + OR (LOP32I x2). Same bits -> same nibble floats, same FMA order -> bit-identical.
__device__ __forceinline__ unsigned unpk(unsigned v) {
    unsigned r;
    asm("lop3.b32 %0, %1, 0x0F0F0F0F, 0x80808080, 0xEA;" : "=r"(r) : "r"(v));
    return r;
}
// EX (MOE2 tail only): the main-loop contraction fma(d0, P, d1 * Q), P/Q = fma(a, 128, -136 * sx) (from PTX+SASS), spelled out
// so a different call site cannot contract the same expression the other way round.
template <bool L3 = false, bool EX = false>
__device__ __forceinline__ float dot2c(const Q9 & q, const float * xc, float2 sx) {
    const unsigned w0 = q.w[0], w1 = q.w[1], w2 = q.w[2], w3 = q.w[3], w4 = q.w[4];
    float d0 = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
    float d1 = __half2float(__ushort_as_half((unsigned short)(w4 >> 16)));
    unsigned q0[4] = { __byte_perm(w0, w1, 0x5432), __byte_perm(w1, w2, 0x5432), __byte_perm(w2, w3, 0x5432), __byte_perm(w3, w4, 0x5432) };
    unsigned q1[4] = { q.w[5], q.w[6], q.w[7], q.w[8] };
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned lo = L3 ? unpk(q0[i]) : (q0[i] & 0x0F0F0F0Fu) | 0x80808080u, hi = L3 ? unpk(q0[i] >> 4) : ((q0[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        float4 xl = *(const float4 *)(xc + 4 * i), xh = *(const float4 *)(xc + 16 + 4 * i);
        a0 = fmaf(nib(lo, 0), xl.x, a0); a0 = fmaf(nib(lo, 1), xl.y, a0); a0 = fmaf(nib(lo, 2), xl.z, a0); a0 = fmaf(nib(lo, 3), xl.w, a0);
        a0 = fmaf(nib(hi, 0), xh.x, a0); a0 = fmaf(nib(hi, 1), xh.y, a0); a0 = fmaf(nib(hi, 2), xh.z, a0); a0 = fmaf(nib(hi, 3), xh.w, a0);
        lo = L3 ? unpk(q1[i]) : (q1[i] & 0x0F0F0F0Fu) | 0x80808080u; hi = L3 ? unpk(q1[i] >> 4) : ((q1[i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        xl = *(const float4 *)(xc + 32 + 4 * i); xh = *(const float4 *)(xc + 48 + 4 * i);
        a1 = fmaf(nib(lo, 0), xl.x, a1); a1 = fmaf(nib(lo, 1), xl.y, a1); a1 = fmaf(nib(lo, 2), xl.z, a1); a1 = fmaf(nib(lo, 3), xl.w, a1);
        a1 = fmaf(nib(hi, 0), xh.x, a1); a1 = fmaf(nib(hi, 1), xh.y, a1); a1 = fmaf(nib(hi, 2), xh.z, a1); a1 = fmaf(nib(hi, 3), xh.w, a1);
    }
    if (EX) return __fmaf_rn(d0, __fmaf_rn(a0, 128.f, -__fmul_rn(sx.x, 136.f)), __fmul_rn(d1, __fmaf_rn(a1, 128.f, -__fmul_rn(sx.y, 136.f))));
    return d0 * (128.f * a0 - 136.f * sx.x) + d1 * (128.f * a1 - 136.f * sx.y);
}
__device__ __forceinline__ void load_norm(float * xs, float2 * sums, const float * h, const float * w, float * red, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += NT) { float v = h[i]; xs[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += red[i];
    const float sc = rsqrtf(tot / H + eps);
    for (int i = threadIdx.x; i < H; i += NT) xs[(i >> 6) * CH + (i & 63)] *= sc * __ldg(w + i);
    __syncthreads();
    for (int b = warp; b < H / 32; b += NW) {
        float s = wsum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
        if (lane == 0) ((float *)sums)[b] = s;
    }
    __syncthreads();
}
__device__ __forceinline__ void ldA(const Layer & L, int t, int lane, float4 * w) {
    const int row = t >> 2, q = t & 3;
    const float4 * p = (const float4 *)((row < NE ? L.gate_inp + (size_t)row * H : L.gate_inp_sh) + q * QK);
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = __ldg(p + lane + 32 * i);
}
__device__ __forceinline__ void rowB(const Layer & L, const int * sel, int t, const uint8_t *& g, const uint8_t *& u) {
    if (t < FFS) { g = L.gate_sh + (size_t)t * ROW_GU; u = L.up_sh + (size_t)t * ROW_GU; }
    else { const int k = (t - FFS) / FF, r = (t - FFS) % FF; const size_t off = ((size_t)sel[k] * FF + r) * ROW_GU; g = L.gate_exps + off; u = L.up_exps + off; }
}
__device__ __forceinline__ void ldB(const uint8_t * g, const uint8_t * u, int lane, uint4 * v) {
    const uint4 * g4 = (const uint4 *)g, * u4 = (const uint4 *)u;
    v[0] = __ldg(g4 + lane); v[1] = __ldg(g4 + lane + 32); v[3] = __ldg(u4 + lane); v[4] = __ldg(u4 + lane + 32);
    if (lane < 8) { v[2] = __ldg(g4 + lane + 64); v[5] = __ldg(u4 + lane + 64); }
}
__device__ __forceinline__ void route(const Scratch * s, int lane, int * sel, float * selw) {
    float v[NE / 32];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) {
        float a = 0.f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) a += __ldcg(&s->lpart[q][lane + 32 * i]);
        v[i] = a; m = fmaxf(m, a);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) { v[i] = expf(v[i] - m); sum += v[i]; }
    sum = wsum(sum);
    const float inv = 1.f / sum;
#pragma unroll
    for (int i = 0; i < NE / 32; ++i) v[i] *= inv;
    float ws = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        float bv = -1.f; int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < NE / 32; ++i) if (v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
#pragma unroll
        for (int i = 0; i < NE / 32; ++i) if (bi == lane + 32 * i) v[i] = -2.f;
        if (lane == 0) { sel[k] = bi; selw[k] = bv; }
        ws += bv;
    }
    ws = fmaxf(ws, 6.103515625e-5f);
    __syncwarp();
    if (lane < TOPK) selw[lane] /= ws;
}

// DX (GGML_CUDA_MOE_DOWNX=1): down prologue reordered so the L2-resident activations (h_exp/h_sh/gpart) are
// requested before the first down rows (DRAM), instead of queueing behind them; group sums straight from
// registers (same lanes, same butterfly) -> one __syncthreads instead of two (bit 1).
// Bit 2: down rows dealt warp-major (row = warp * grid + block + k * W) so the partial last row wave
// (2048 rows over 384 warps) is spread over every block/SM instead of the first 16 blocks.
// Rows are independent and each row's arithmetic is unchanged -> bit-identical output.
// C07 (GGML_CUDA_MOE_DOWNH, default off; fused PHASE 0 only):
// bit 4 (DOWNH=1): selected-expert down row bases cb[] (sel valid since route+__syncthreads in phase B) and the
//   shexp gate gpart -> gl -> sigmoid (gpart final since barrier 1) are formed before barrier 2, not after it.
// bit 8 (DOWNH=2, needs bit 4): the first down row of each warp (read-only weights, DRAM) is also issued before
//   barrier 2, so its latency overlaps the barrier wait instead of the activation prologue.
// No added global traffic, no added barrier/__syncthreads; same loads, same sum order, same FP32 ops.
template <int PHASE, int DX = 0>
__global__ void __launch_bounds__(NT, 2) moe_v2(Params P) {
    extern __shared__ float smem[];
    __shared__ int sel[TOPK];
    __shared__ float selw[TOPK];
    __shared__ float red[NW];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + (TOPK * FF + FFS) / 64 * CH);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + warp * STW;
    uint4 * stg4 = (uint4 *)stg;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    Scratch * s = P.s;
    const Layer & L = P.L;

    if (PHASE == 0 || PHASE == 1) {
        float4 cw[4], nw[4];
        int t = gw;
        if (t < NTA) ldA(L, t, lane, cw);
        load_norm(xs, sums, P.h, P.norm, red, P.eps);
        for (; t < NTA; t += W) {
            const int tn = t + W;
            if (tn < NTA) ldA(L, tn, lane, nw);
            const int row = t >> 2, q = t & 3;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int e = q * (QK / 4) + lane + 32 * i;
                const float4 xv = *(const float4 *)(xs + ((4 * e) >> 6) * CH + ((4 * e) & 63));
                a = fmaf(cw[i].x, xv.x, a); a = fmaf(cw[i].y, xv.y, a); a = fmaf(cw[i].z, xv.z, a); a = fmaf(cw[i].w, xv.w, a);
            }
            a = wsum(a);
            if (lane == 0) { if (row < NE) s->lpart[q][row] = a; else s->gpart[q] = a; }
#pragma unroll
            for (int i = 0; i < 4; ++i) cw[i] = nw[i];
        }
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 2) {
        uint4 cv[6];
        bool have = false;
        const uint8_t * g, * u;
        if (gw < FFS) { rowB(L, sel, gw, g, u); ldB(g, u, lane, cv); have = true; }
        if (PHASE == 2) load_norm(xs, sums, P.h, P.norm, red, P.eps);
        if (warp == 0) route(s, lane, sel, selw);
        __syncthreads();
        const float * xc = xs + lane * CH; const float2 sx = sums[lane];
        for (int t = gw; t < NTB; t += W) {
            if (!have) { rowB(L, sel, t, g, u); ldB(g, u, lane, cv); }
            stg4[lane] = cv[0]; stg4[lane + 32] = cv[1]; stg4[72 + lane] = cv[3]; stg4[104 + lane] = cv[4];
            if (lane < 8) { stg4[lane + 64] = cv[2]; stg4[136 + lane] = cv[5]; }
            __syncwarp();
            const int tn = t + W; have = tn < NTB;
            if (have) { rowB(L, sel, tn, g, u); ldB(g, u, lane, cv); }
            float gg = wsum(dot2c<(DX & 16) != 0>(lds9(stg + lane * 9), xc, sx));
            float uu = wsum(dot2c<(DX & 16) != 0>(lds9(stg + 288 + lane * 9), xc, sx));
            if (lane == 0) { if (t < FFS) s->h_sh[t] = silu(gg) * uu; else s->h_exp[t - FFS] = silu(gg) * uu; }
            __syncwarp();
        }
    }
    constexpr bool HS = PHASE == 0 && (DX & 4), PF = HS && (DX & 8);
    const uint8_t * cb[6];
    uint4 cv[6];
    float gl = 0.f;
    int r = 0;
    if (HS) {
        r = (DX & 2) ? warp * gridDim.x + blockIdx.x : gw;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int uidx = lane + 32 * i, m = uidx / 18, w = uidx % 18;
            cb[i] = (m < TOPK ? L.down_exps + (size_t)sel[m < TOPK ? m : 0] * H * ROW_D : L.down_sh) + w * 16;
        }
        if (PF && r < H) {
#pragma unroll
            for (int i = 0; i < 5; ++i) cv[i] = __ldg((const uint4 *)(cb[i] + (size_t)r * ROW_D));
            if (lane < 2) cv[5] = __ldg((const uint4 *)(cb[5] + (size_t)r * ROW_D));
        }
        float gp[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) gp[q] = __ldcg(&s->gpart[q]);
#pragma unroll
        for (int q = 0; q < NQ; ++q) gl += gp[q];
    }
    if (PHASE == 0) grid_barrier(s);
    if (PHASE == 0 || PHASE == 3) {
        if (PHASE == 3) { if (warp == 0) route(s, lane, sel, selw); __syncthreads(); }
        constexpr int NX = (TOPK * FF + FFS) / NT, NXE = TOPK * FF / NT;
        float hv[(DX & 1) ? NX : 1], gp[NQ];
        if (DX & 1) {
#pragma unroll
            for (int j = 0; j < NX; ++j) {
                const int i = threadIdx.x + NT * j;
                hv[j] = j < NXE ? __ldcg(&s->h_exp[i]) : __ldcg(&s->h_sh[i - TOPK * FF]);
            }
#pragma unroll
            if (!HS) for (int q = 0; q < NQ; ++q) gp[q] = __ldcg(&s->gpart[q]);
        }
        if (!HS) {
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int uidx = lane + 32 * i, m = uidx / 18, w = uidx % 18;
            cb[i] = (m < TOPK ? L.down_exps + (size_t)sel[m < TOPK ? m : 0] * H * ROW_D : L.down_sh) + w * 16;
        }
        }
        if (!HS) r = (DX & 2) ? warp * gridDim.x + blockIdx.x : gw;
        if (!PF && r < H) {
#pragma unroll
            for (int i = 0; i < 5; ++i) cv[i] = __ldg((const uint4 *)(cb[i] + (size_t)r * ROW_D));
            if (lane < 2) cv[5] = __ldg((const uint4 *)(cb[5] + (size_t)r * ROW_D));
        }
        if (DX & 1) {
            // thread (warp, lane) slot j holds element 256j + 32warp + lane = group 8j + warp, lane position lane
#pragma unroll
            for (int j = 0; j < NX; ++j) {
                const int i = threadIdx.x + NT * j;
                xs[(i >> 6) * CH + (i & 63)] = hv[j];
                const float sm = wsum(hv[j]);
                if (lane == 0) ((float *)sums)[NW * j + warp] = sm;
            }
            __syncthreads();
            if (!HS) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) gl += gp[q];
            }
        } else {
        for (int i = threadIdx.x; i < TOPK * FF + FFS; i += NT) {
            float val = i < TOPK * FF ? __ldcg(&s->h_exp[i]) : __ldcg(&s->h_sh[i - TOPK * FF]);
            xs[(i >> 6) * CH + (i & 63)] = val;
        }
        __syncthreads();
        for (int b = warp; b < (TOPK * FF + FFS) / 32; b += NW) {
            float sm = wsum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
            if (lane == 0) ((float *)sums)[b] = sm;
        }
        __syncthreads();
            if (!HS) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) gl += __ldcg(&s->gpart[q]);
            }
        }
        const float gs = 1.f / (1.f + expf(-gl));
        const float w0 = selw[lane >> 3], w1 = selw[(lane >> 3) + 4];
        // MOE2 bit 32 (GGML_CUDA_MOE_TAILBAL): rows >= HF (the partial last wave) are dealt block-wise instead of to
        // the first blocks' warps; their loads take the last main-loop prefetch slot (same bytes, no extra traffic).
        constexpr bool TB = (DX & 32) != 0;
#ifndef MOE2_DBG
#define MOE2_DBG 0
#endif
        const int HF = TB && !MOE2_DBG ? (H / W) * W : H;
        const int TT = H - (H / W) * W, tper = TT / (int)gridDim.x, tex = TT % (int)gridDim.x, bx = blockIdx.x;
        const int tn = TB ? tper + (bx < tex) : 0, tbase = (H / W) * W + bx * tper + min(bx, tex);
        for (; r < HF; r += W) {
#pragma unroll
            for (int i = 0; i < 5; ++i) stg4[lane + 32 * i] = cv[i];
            if (lane < 2) stg4[160 + lane] = cv[5];
            __syncwarp();
            const int rn = r + W;
            if (rn < HF) {
#pragma unroll
                for (int i = 0; i < 5; ++i) cv[i] = __ldg((const uint4 *)(cb[i] + (size_t)rn * ROW_D));
                if (lane < 2) cv[5] = __ldg((const uint4 *)(cb[5] + (size_t)rn * ROW_D));
            } else if (TB) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int k = threadIdx.x + NT * i;
                    if (k < tn * 162) {
                        const int j = k / 162, u = k % 162, m = u / 18, w = u % 18;
                        const uint8_t * b = (m < TOPK ? L.down_exps + (size_t)sel[m < TOPK ? m : 0] * H * ROW_D : L.down_sh) + w * 16;
                        cv[i] = __ldg((const uint4 *)(b + (size_t)(tbase + j) * ROW_D));
                    }
                }
            }
            float acc = w0 * dot2c<(DX & 16) != 0>(lds9(stg + lane * 9), xs + lane * CH, sums[lane])
                      + w1 * dot2c<(DX & 16) != 0>(lds9(stg + (lane + 32) * 9), xs + (lane + 32) * CH, sums[lane + 32]);
            if (lane < 8) acc += gs * dot2c<(DX & 16) != 0>(lds9(stg + (lane + 64) * 9), xs + (lane + 64) * CH, sums[lane + 64]);
            acc = wsum(acc);
            if (lane == 0) P.out[r] = P.h[r] + acc;
            __syncwarp();
        }
        if (TB && MOE2_DBG) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int k = threadIdx.x + NT * i;
                    if (k < tn * 162) {
                        const int j = k / 162, u = k % 162, m = u / 18, w = u % 18;
                        const uint8_t * b = (m < TOPK ? L.down_exps + (size_t)sel[m < TOPK ? m : 0] * H * ROW_D : L.down_sh) + w * 16;
                        cv[i] = __ldg((const uint4 *)(b + (size_t)(tbase + j) * ROW_D));
                    }
                }
        }
        if (TB && tn > 0) {   // tn is block-uniform -> barriers are uniform
            unsigned * bst = (unsigned *)(smem + SMEM_FLOATS);
            float * dts = (float *)(bst + 3 * STW);
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 2; ++i) { const int k = threadIdx.x + NT * i; if (k < tn * 162) ((uint4 *)bst)[k] = cv[i]; }
            __syncthreads();
            if (threadIdx.x < tn * 72) {
                const int j = threadIdx.x / 72, q = threadIdx.x % 72;
                dts[threadIdx.x] = dot2c<(DX & 16) != 0, true>(lds9(bst + j * STW + q * 9), xs + q * CH, sums[q]);
            }
            __syncthreads();
            if (warp < tn) {
                const float * dw = dts + warp * 72;
                float acc = __fmaf_rn(w0, dw[lane], __fmul_rn(w1, dw[lane + 32]));
                if (lane < 8) acc = __fmaf_rn(gs, dw[lane + 64], acc);
                acc = wsum(acc);
                if (lane == 0) P.out[tbase + warp] = P.h[tbase + warp] + acc;
            }
        }
    }
}

// WP4 moe_v3 (GGML_CUDA_MOE_V3=1, default off): occupancy rework of moe_v2<0, 61>.
// 512-thread blocks share one activation copy (xs/sums) across 16 warps, and each warp stages 1152 B instead of
// 2592 B: a gate/up row is staged as two 72-uint4 chunks (gate, then up), a down row as three chunks (experts 0-3,
// experts 4-7, shared expert). Target: <=64 regs, ~39 KB smem -> 2 x 512 threads = 32 warps/SM (was 16).
// Bit-identity with moe_v2<0, 61>: every row keeps its lane -> unit mapping, dot2c form (LOP3), butterfly order
// and the EX combine of the MOE2 tail; load_norm keeps the 256-thread sum-of-squares split; activation group sums
// are per 32-element group as before. Only the grid / warp -> row assignment and the staging granularity change.
constexpr int NT3 = 512, NW3 = NT3 / 32;
constexpr int STW3 = 72 * 4;
constexpr int SMEM3_FLOATS = SMEM_FLOATS + NW3 * STW3;

__device__ __forceinline__ void load_norm3(float * xs, float2 * sums, const float * h, const float * w, float * red, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x < NT) {   // the first 256 threads reproduce moe_v2's per-thread / per-warp sum-of-squares split
        float ss = 0.f;
        for (int i = threadIdx.x; i < H; i += NT) { float v = h[i]; xs[(i >> 6) * CH + (i & 63)] = v; ss += v * v; }
        ss = wsum(ss);
        if (lane == 0) red[warp] = ss;
    }
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += red[i];
    const float sc = rsqrtf(tot / H + eps);
    for (int i = threadIdx.x; i < H; i += NT3) xs[(i >> 6) * CH + (i & 63)] *= sc * __ldg(w + i);
    __syncthreads();
    for (int b = warp; b < H / 32; b += NW3) {
        float s = wsum(xs[(b >> 1) * CH + (b & 1) * 32 + lane]);
        if (lane == 0) ((float *)sums)[b] = s;
    }
    __syncthreads();
}
// one 72-uint4 chunk (a 1152 B gate or up row) per warp: lane holds uint4 lane, lane+32 and (lane < 8) lane+64
__device__ __forceinline__ void ld72(const uint8_t * p, int lane, uint4 * v) {
    const uint4 * p4 = (const uint4 *)p;
    v[0] = __ldg(p4 + lane); v[1] = __ldg(p4 + lane + 32);
    if (lane < 8) v[2] = __ldg(p4 + lane + 64);
}
__device__ __forceinline__ void st72(uint4 * stg4, int lane, const uint4 * v) {
    stg4[lane] = v[0]; stg4[lane + 32] = v[1];
    if (lane < 8) stg4[lane + 64] = v[2];
}
// down chunk c of row r: c < 2 -> selected experts 4c..4c+3 (72 uint4, = moe_v2 units 32c..32c+31),
// c == 2 -> shared-expert row (18 uint4 in lanes < 18, = moe_v2 units 64..71)
__device__ __forceinline__ void ldD(const Layer & L, const int * sel, int r, int c, int lane, uint4 * v) {
    if (c < 2) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int k = lane + 32 * i;
            if (i < 2 || lane < 8) {
                const int m = 4 * c + k / 18, w = k % 18;
                v[i] = __ldg((const uint4 *)(L.down_exps + ((size_t)sel[m] * H + r) * ROW_D) + w);
            }
        }
    } else if (lane < 18) {
        v[0] = __ldg((const uint4 *)(L.down_sh + (size_t)r * ROW_D) + lane);
    }
}

template <bool L3>
__global__ void __launch_bounds__(NT3, 2) moe_v3(Params P) {
    extern __shared__ float smem[];
    __shared__ int sel[TOPK];
    __shared__ float selw[TOPK];
    __shared__ float red[NW];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + (TOPK * FF + FFS) / 64 * CH);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + warp * STW3;
    uint4 * stg4 = (uint4 *)stg;
    const int gw = blockIdx.x * NW3 + warp, W = gridDim.x * NW3;
    Scratch * s = P.s;
    const Layer & L = P.L;

    // A: rmsnorm + router logits / shexp-gate quarter-row partials (per task identical to moe_v2)
    {
        float4 cw[4];
        int t = gw;
        if (t < NTA) ldA(L, t, lane, cw);
        load_norm3(xs, sums, P.h, P.norm, red, P.eps);
        for (; t < NTA; t += W) {
            const int row = t >> 2, q = t & 3;
            float a = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int e = q * (QK / 4) + lane + 32 * i;
                const float4 xv = *(const float4 *)(xs + ((4 * e) >> 6) * CH + ((4 * e) & 63));
                a = fmaf(cw[i].x, xv.x, a); a = fmaf(cw[i].y, xv.y, a); a = fmaf(cw[i].z, xv.z, a); a = fmaf(cw[i].w, xv.w, a);
            }
            a = wsum(a);
            if (lane == 0) { if (row < NE) s->lpart[q][row] = a; else s->gpart[q] = a; }
            if (t + W < NTA) ldA(L, t + W, lane, cw);
        }
    }
    grid_barrier(s);
    // B: route; gate row chunk, then up row chunk, per row
    {
        uint4 cv[3];
        bool have = false;
        const uint8_t * g, * u;
        if (gw < FFS) { rowB(L, sel, gw, g, u); ld72(g, lane, cv); have = true; }
        if (warp == 0) route(s, lane, sel, selw);
        __syncthreads();
        const float * xc = xs + lane * CH; const float2 sx = sums[lane];
        for (int t = gw; t < NTB; t += W) {
            if (!have) { rowB(L, sel, t, g, u); ld72(g, lane, cv); }
            st72(stg4, lane, cv);
            __syncwarp();
            ld72(u, lane, cv);
            const float gg = wsum(dot2c<L3>(lds9(stg + lane * 9), xc, sx));
            __syncwarp();
            st72(stg4, lane, cv);
            __syncwarp();
            const int tn = t + W; have = tn < NTB;
            if (have) { rowB(L, sel, tn, g, u); ld72(g, lane, cv); }
            const float uu = wsum(dot2c<L3>(lds9(stg + lane * 9), xc, sx));
            if (lane == 0) { if (t < FFS) s->h_sh[t] = silu(gg) * uu; else s->h_exp[t - FFS] = silu(gg) * uu; }
            __syncwarp();
        }
    }
    // C: down. First chunk of each warp's first row and the shexp gate logit are issued before barrier 2 (as DOWNH=2).
    int r = warp * gridDim.x + blockIdx.x;   // warp-major rows: the partial last wave is spread over every block
    uint4 cv[3];
    if (r < H) ldD(L, sel, r, 0, lane, cv);
    float gl = 0.f;
    {
        float gp[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) gp[q] = __ldcg(&s->gpart[q]);
#pragma unroll
        for (int q = 0; q < NQ; ++q) gl += gp[q];
    }
    grid_barrier(s);
    {
        constexpr int NX = (TOPK * FF + FFS) / NT3, NXE = TOPK * FF / NT3;
        float hv[NX];
#pragma unroll
        for (int j = 0; j < NX; ++j) {
            const int i = threadIdx.x + NT3 * j;
            hv[j] = j < NXE ? __ldcg(&s->h_exp[i]) : __ldcg(&s->h_sh[i - TOPK * FF]);
        }
        // thread (warp, lane) slot j holds element 512j + 32warp + lane = 32-element group 16j + warp
#pragma unroll
        for (int j = 0; j < NX; ++j) {
            const int i = threadIdx.x + NT3 * j;
            xs[(i >> 6) * CH + (i & 63)] = hv[j];
            const float sm = wsum(hv[j]);
            if (lane == 0) ((float *)sums)[NW3 * j + warp] = sm;
        }
        __syncthreads();
    }
    const float gs = 1.f / (1.f + expf(-gl));
    const float w0 = selw[lane >> 3], w1 = selw[(lane >> 3) + 4];
    for (; r < H; r += W) {
        st72(stg4, lane, cv);
        __syncwarp();
        ldD(L, sel, r, 1, lane, cv);
        const float dA = dot2c<L3, true>(lds9(stg + lane * 9), xs + lane * CH, sums[lane]);
        __syncwarp();
        st72(stg4, lane, cv);
        __syncwarp();
        ldD(L, sel, r, 2, lane, cv);
        const float dB = dot2c<L3, true>(lds9(stg + lane * 9), xs + (lane + 32) * CH, sums[lane + 32]);
        __syncwarp();
        if (lane < 18) stg4[lane] = cv[0];
        __syncwarp();
        if (r + W < H) ldD(L, sel, r + W, 0, lane, cv);
        float acc = __fmaf_rn(w0, dA, __fmul_rn(w1, dB));
        if (lane < 8) acc = __fmaf_rn(gs, dot2c<L3, true>(lds9(stg + lane * 9), xs + (lane + 64) * CH, sums[lane + 64]), acc);
        acc = wsum(acc);
        if (lane == 0) P.out[r] = P.h[r] + acc;
        __syncwarp();
    }
}
__global__ void moe_v3_cmp(const float * a, const float * b, unsigned * res) {
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const unsigned x = __float_as_uint(a[i]), y = __float_as_uint(b[i]);
        if (x != y) { atomicAdd(&res[0], 1u); atomicMax(&res[1], __float_as_uint(fabsf(a[i] - b[i]))); }
    }
}
