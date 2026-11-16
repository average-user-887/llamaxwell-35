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
// 28h (GGML_CUDA_GDN2_A2ROW=1, default off): dot2v2 for two weight rows (u, v) sharing each smem activation load.
// Each of the four accumulator chains is the same fmaf sequence in the same order as dot2v2 -> bit-identical per row.
__device__ __forceinline__ void dot2v2x2(const unsigned (&u)[9], const unsigned (&v)[9], const float * xc, float2 sx,
                                         const float * yc, float2 sy, float & ux, float & uy, float & vx, float & vy) {
    const float ud0 = __half2float(__ushort_as_half((unsigned short)(u[0] & 0xFFFF))), ud1 = __half2float(__ushort_as_half((unsigned short)(u[4] >> 16)));
    const float vd0 = __half2float(__ushort_as_half((unsigned short)(v[0] & 0xFFFF))), vd1 = __half2float(__ushort_as_half((unsigned short)(v[4] >> 16)));
    float ua0 = 0.f, ua1 = 0.f, ub0 = 0.f, ub1 = 0.f, va0 = 0.f, va1 = 0.f, vb0 = 0.f, vb1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned uq0 = __byte_perm(u[i], u[i + 1], 0x5432), vq0 = __byte_perm(v[i], v[i + 1], 0x5432);
        unsigned ulo = (uq0 & 0x0F0F0F0Fu) | 0x80808080u, uhi = ((uq0 >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        unsigned vlo = (vq0 & 0x0F0F0F0Fu) | 0x80808080u, vhi = ((vq0 >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        float4 xl = *(const float4 *)(xc + 4 * i), xh = *(const float4 *)(xc + 16 + 4 * i);
        ua0 = fmaf(nib(ulo, 0), xl.x, ua0); ua0 = fmaf(nib(ulo, 1), xl.y, ua0); ua0 = fmaf(nib(ulo, 2), xl.z, ua0); ua0 = fmaf(nib(ulo, 3), xl.w, ua0);
        ua0 = fmaf(nib(uhi, 0), xh.x, ua0); ua0 = fmaf(nib(uhi, 1), xh.y, ua0); ua0 = fmaf(nib(uhi, 2), xh.z, ua0); ua0 = fmaf(nib(uhi, 3), xh.w, ua0);
        va0 = fmaf(nib(vlo, 0), xl.x, va0); va0 = fmaf(nib(vlo, 1), xl.y, va0); va0 = fmaf(nib(vlo, 2), xl.z, va0); va0 = fmaf(nib(vlo, 3), xl.w, va0);
        va0 = fmaf(nib(vhi, 0), xh.x, va0); va0 = fmaf(nib(vhi, 1), xh.y, va0); va0 = fmaf(nib(vhi, 2), xh.z, va0); va0 = fmaf(nib(vhi, 3), xh.w, va0);
        xl = *(const float4 *)(yc + 4 * i); xh = *(const float4 *)(yc + 16 + 4 * i);
        ub0 = fmaf(nib(ulo, 0), xl.x, ub0); ub0 = fmaf(nib(ulo, 1), xl.y, ub0); ub0 = fmaf(nib(ulo, 2), xl.z, ub0); ub0 = fmaf(nib(ulo, 3), xl.w, ub0);
        ub0 = fmaf(nib(uhi, 0), xh.x, ub0); ub0 = fmaf(nib(uhi, 1), xh.y, ub0); ub0 = fmaf(nib(uhi, 2), xh.z, ub0); ub0 = fmaf(nib(uhi, 3), xh.w, ub0);
        vb0 = fmaf(nib(vlo, 0), xl.x, vb0); vb0 = fmaf(nib(vlo, 1), xl.y, vb0); vb0 = fmaf(nib(vlo, 2), xl.z, vb0); vb0 = fmaf(nib(vlo, 3), xl.w, vb0);
        vb0 = fmaf(nib(vhi, 0), xh.x, vb0); vb0 = fmaf(nib(vhi, 1), xh.y, vb0); vb0 = fmaf(nib(vhi, 2), xh.z, vb0); vb0 = fmaf(nib(vhi, 3), xh.w, vb0);
        ulo = (u[5 + i] & 0x0F0F0F0Fu) | 0x80808080u; uhi = ((u[5 + i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        vlo = (v[5 + i] & 0x0F0F0F0Fu) | 0x80808080u; vhi = ((v[5 + i] >> 4) & 0x0F0F0F0Fu) | 0x80808080u;
        xl = *(const float4 *)(xc + 32 + 4 * i); xh = *(const float4 *)(xc + 48 + 4 * i);
        ua1 = fmaf(nib(ulo, 0), xl.x, ua1); ua1 = fmaf(nib(ulo, 1), xl.y, ua1); ua1 = fmaf(nib(ulo, 2), xl.z, ua1); ua1 = fmaf(nib(ulo, 3), xl.w, ua1);
        ua1 = fmaf(nib(uhi, 0), xh.x, ua1); ua1 = fmaf(nib(uhi, 1), xh.y, ua1); ua1 = fmaf(nib(uhi, 2), xh.z, ua1); ua1 = fmaf(nib(uhi, 3), xh.w, ua1);
        va1 = fmaf(nib(vlo, 0), xl.x, va1); va1 = fmaf(nib(vlo, 1), xl.y, va1); va1 = fmaf(nib(vlo, 2), xl.z, va1); va1 = fmaf(nib(vlo, 3), xl.w, va1);
        va1 = fmaf(nib(vhi, 0), xh.x, va1); va1 = fmaf(nib(vhi, 1), xh.y, va1); va1 = fmaf(nib(vhi, 2), xh.z, va1); va1 = fmaf(nib(vhi, 3), xh.w, va1);
        xl = *(const float4 *)(yc + 32 + 4 * i); xh = *(const float4 *)(yc + 48 + 4 * i);
        ub1 = fmaf(nib(ulo, 0), xl.x, ub1); ub1 = fmaf(nib(ulo, 1), xl.y, ub1); ub1 = fmaf(nib(ulo, 2), xl.z, ub1); ub1 = fmaf(nib(ulo, 3), xl.w, ub1);
        ub1 = fmaf(nib(uhi, 0), xh.x, ub1); ub1 = fmaf(nib(uhi, 1), xh.y, ub1); ub1 = fmaf(nib(uhi, 2), xh.z, ub1); ub1 = fmaf(nib(uhi, 3), xh.w, ub1);
        vb1 = fmaf(nib(vlo, 0), xl.x, vb1); vb1 = fmaf(nib(vlo, 1), xl.y, vb1); vb1 = fmaf(nib(vlo, 2), xl.z, vb1); vb1 = fmaf(nib(vlo, 3), xl.w, vb1);
        vb1 = fmaf(nib(vhi, 0), xh.x, vb1); vb1 = fmaf(nib(vhi, 1), xh.y, vb1); vb1 = fmaf(nib(vhi, 2), xh.z, vb1); vb1 = fmaf(nib(vhi, 3), xh.w, vb1);
    }
    ux = ud0 * (128.f * ua0 - 136.f * sx.x) + ud1 * (128.f * ua1 - 136.f * sx.y);
    uy = ud0 * (128.f * ub0 - 136.f * sy.x) + ud1 * (128.f * ub1 - 136.f * sy.y);
    vx = vd0 * (128.f * va0 - 136.f * sx.x) + vd1 * (128.f * va1 - 136.f * sx.y);
    vy = vd0 * (128.f * vb0 - 136.f * sy.x) + vd1 * (128.f * vb1 - 136.f * sy.y);
}
struct Params2 { Params p[2]; };
constexpr int XS_HALF = H / 64 * CH;   // 2176 floats: one K=2048 activation set
constexpr int N2R = 4;                 // max ssm_out rows per warp
// Card 18 (GGML_CUDA_OCCAMY_GDN18=1, default OFF): IL=1 runs phase B for both sequences per item (one item loop,
// both state tiles loaded back to back, one set of block barriers per item) instead of sequence 0 then sequence 1.
// Per sequence every value is the same expression in the same order as IL=0, so the output is bit-identical.
// Card A3 (GGML_CUDA_OCCAMY_GDN_A3=1|2, default OFF): IL=2 keeps the card-18 item loop and interleaves the four
// delta-rule chains of an item (2 sequences x 2 columns): the 4 k.S dots are reduced in one interleaved butterfly,
// then the 4 q.S' dots, so the 10 shuffle levels of each chain overlap instead of running per sequence. The per-item
// ssq reduction goes to a parity double-buffered smem array, which drops 2 of the 4 block barriers per item.
// IL=3 adds the next item's state tiles (both sequences) loaded one item ahead. Same expressions, same order: bit-identical.
template <int A2 = 0, int IL = 0>   // bundle9: A2 = card 28h phase-A row pairs, IL = card 18 phase-B interleave (independent)

__global__ void __launch_bounds__(NT, 3) gdn_fused2(const Params2 PP) {
    extern __shared__ float smem[];
    __shared__ float qn[SD], kn[SD], vv[CGC], red[NW];
    __shared__ unsigned hc[NVH], fin[NVH + 1];
    float * xs = smem;
    float2 * sums = (float2 *)(smem + DI / 64 * CH);
    unsigned * stg = (unsigned *)(smem + SMEM_FLOATS) + (threadIdx.x >> 5) * STW_RP; uint4 * stg4 = (uint4 *)stg;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * NW + warp, W = gridDim.x * NW;
    const Layer & L = PP.p[0].L;
    Scratch * const s0 = PP.p[0].s, * const s1 = PP.p[1].s;

    {   // ---- phase A ----
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
        if (A2) {
        // 28h: the warp's rows gw, gw+W, gw+2W, ... taken in pairs (t, t+W); both rows staged through the same per-warp
        // stage buffer into registers, then dotted together so every smem activation float4 serves two rows.
        uint4 e0, e1, e2;
        if (t < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
        if (t + W < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, t + W); e0 = __ldg(p + lane); e1 = __ldg(p + lane + 32); if (lane < 8) e2 = __ldg(p + lane + 64); }
        for (; t < NROWS_A; t += 2 * W) {
            const int tv = t + W;
            unsigned u[9], v[9];
            stg4[lane] = c0; stg4[lane + 32] = c1; if (lane < 8) stg4[lane + 64] = c2;
            __syncwarp();
#pragma unroll
            for (int j = 0; j < 9; ++j) u[j] = stg[lane * 9 + j];
            __syncwarp();
            if (tv < NROWS_A) {
                stg4[lane] = e0; stg4[lane + 32] = e1; if (lane < 8) stg4[lane + 64] = e2;
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 9; ++j) v[j] = stg[lane * 9 + j];
                __syncwarp();
            } else {
#pragma unroll
                for (int j = 0; j < 9; ++j) v[j] = 0u;
            }
            const int tn = t + 2 * W;
            if (tn < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn); c0 = __ldg(p + lane); c1 = __ldg(p + lane + 32); if (lane < 8) c2 = __ldg(p + lane + 64); }
            if (tn + W < NROWS_A) { const uint4 * p = (const uint4 *)rowA(L, tn + W); e0 = __ldg(p + lane); e1 = __ldg(p + lane + 32); if (lane < 8) e2 = __ldg(p + lane + 64); }
            float a0, a1, b0, b1;
            dot2v2x2(u, v, xc0, sx0, xc1, sx1, a0, a1, b0, b1);
            a0 = warp_sum(a0); a1 = warp_sum(a1); b0 = warp_sum(b0); b1 = warp_sum(b1);
            if (lane == 0) {
                float * d0 = t < NQKV ? &s0->qkv[t] : t < NQKV + DI ? &s0->z[t - NQKV] : &s0->ab[t - NQKV - DI];
                float * d1 = t < NQKV ? &s1->qkv[t] : t < NQKV + DI ? &s1->z[t - NQKV] : &s1->ab[t - NQKV - DI];
                *d0 = a0; *d1 = a1;
                if (tv < NROWS_A) {
                    float * f0 = tv < NQKV ? &s0->qkv[tv] : tv < NQKV + DI ? &s0->z[tv - NQKV] : &s0->ab[tv - NQKV - DI];
                    float * f1 = tv < NQKV ? &s1->qkv[tv] : tv < NQKV + DI ? &s1->z[tv - NQKV] : &s1->ab[tv - NQKV - DI];
                    *f0 = b0; *f1 = b1;
                }
            }
        }
        } else {
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
    }
    grid_barrier(s0);
#pragma unroll
    for (int k = 0; k < (IL ? 0 : 2); ++k) {   // ---- phase B (RQK + PCS), one sequence at a time ----
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
    if (IL) {   // ---- phase B (RQK + PCS), card 18: both sequences per item ----
        // float4 reads of qn1/kn1: static __shared__ arrays are only 4-byte aligned by default (GPU gate 15 Nov: misaligned address at +0x564)
        __shared__ __align__(16) float qn1[SD], kn1[SD], vv1[CGC], red1[NW];
        __shared__ unsigned hc1[NVH], fin1[NVH + 1];
        constexpr bool XI = IL >= 2, PFN = IL == 3;   // card A3: chain interleave + 2 barriers/item; next-item state prefetch
        __shared__ float sq2[XI ? 2 : 1][2][NW];      // A3: per-item ssq partials, parity double-buffered
        int par = 0;
        float * const QN[2] = { qn, qn1 }, * const KN[2] = { kn, kn1 }, * const VV[2] = { vv, vv1 }, * const RD[2] = { red, red1 };
        unsigned * const HC[2] = { hc, hc1 }, * const FN[2] = { fin, fin1 };
        const float * CI[2], * SI[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int row = __ldg(PP.p[k].s_copy);
            CI[k] = PP.p[k].conv_base + (size_t)row * PP.p[k].conv_stride;
            SI[k] = PP.p[k].ssm_base  + (size_t)row * PP.p[k].ssm_stride;
        }
        if (threadIdx.x < NVH) { hc[threadIdx.x] = 0; hc1[threadIdx.x] = 0; if (threadIdx.x == 0) { fin[NVH] = 0; fin1[NVH] = 0; } }
        const int it_lo = (int)(((long long)blockIdx.x * (NVH * NCG)) / gridDim.x);
        const int it_hi = (int)(((long long)(blockIdx.x + 1) * (NVH * NCG)) / gridDim.x);
        int kh_c = -1, hv_c = -1;
        float qs[2] = { 0.f, 0.f }, ks[2] = { 0.f, 0.f }, gexp[2] = { 0.f, 0.f }, beta[2] = { 0.f, 0.f };
        float4 stn[PFN ? 2 : 1][2];   // A3 IL=3: next item's state tiles
        if constexpr (PFN) {
            if (it_lo < it_hi) {
                const int hv = it_lo / (2 * NCG) + NKH * ((it_lo / NCG) & 1), col0 = (it_lo % NCG) * CGC + warp * 2;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float4 * si = (const float4 *)(SI[k] + ((size_t)hv * SD + col0) * SD) + lane;
                    stn[k][0] = si[0]; stn[k][1] = si[SD / 4];
                }
            }
        }
        for (int it = it_lo; it < it_hi; ++it) {
            const int hv = it / (2 * NCG) + NKH * ((it / NCG) & 1);
            const int cg = it % NCG, kh = hv % NKH;
            const bool need_qk = kh != kh_c;
            const int col0 = cg * CGC + warp * 2;
            float4 st[2][2];
            if constexpr (PFN) {   // state rows are disjoint per item, so the in-place write of item it cannot alias item it+1
#pragma unroll
                for (int k = 0; k < 2; ++k) { st[k][0] = stn[k][0]; st[k][1] = stn[k][1]; }
                if (it + 1 < it_hi) {
                    const int hn = (it + 1) / (2 * NCG) + NKH * (((it + 1) / NCG) & 1), cn = ((it + 1) % NCG) * CGC + warp * 2;
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const float4 * si = (const float4 *)(SI[k] + ((size_t)hn * SD + cn) * SD) + lane;
                        stn[k][0] = si[0]; stn[k][1] = si[SD / 4];
                    }
                }
            } else {
#pragma unroll
            for (int k = 0; k < 2; ++k) {   // state tiles first: they do not depend on this item's conv/norm
                const float4 * si = (const float4 *)(SI[k] + ((size_t)hv * SD + col0) * SD) + lane;
                st[k][0] = si[0]; st[k][1] = si[SD / 4];
            }
            }
            {
                const int t = threadIdx.x;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const Params & P = PP.p[k];
                    if (need_qk) {
                    const int c = t < SD ? kh * SD + t : H + kh * SD + (t - SD);
                    const float * ci = CI[k] + c * 3; const float * w = L.conv_w + c * 4;
                    float xnew = __ldcg(&P.s->qkv[c]);
                    float y = ci[0] * w[0] + ci[1] * w[1] + ci[2] * w[2] + xnew * w[3];
                    y = silu(y);
                    float sq = warp_sum(y * y);
                    if (lane == 0) RD[k][warp] = sq;
                    if (t < SD) QN[k][t] = y; else KN[k][t - SD] = y;
                    }
                    if (t < CGC) {
                        const int cv = 2 * H + hv * SD + cg * CGC + t;
                        const float * cvi = CI[k] + cv * 3; const float * wv = L.conv_w + cv * 4;
                        float xv = __ldcg(&P.s->qkv[cv]);
                        float c1 = cvi[1], c2 = cvi[2];
                        float yv = cvi[0] * wv[0] + c1 * wv[1] + c2 * wv[2] + xv * wv[3];
                        VV[k][t] = silu(yv);
                        float * co = P.conv_out + cv * 3; co[0] = c1; co[1] = c2; co[2] = xv;
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const Params & P = PP.p[k];
                if (need_qk) {
                qs[k] = rsqrtf((RD[k][0] + RD[k][1] + RD[k][2] + RD[k][3]) / SD + P.eps_l2) * P.scale_l2;
                ks[k] = rsqrtf((RD[k][4] + RD[k][5] + RD[k][6] + RD[k][7]) / SD + P.eps_l2) * P.scale_l2;
                }
                if (hv != hv_c) {
                const float al = __ldcg(&P.s->ab[hv]) + __ldg(L.dt_bias + hv);
                const float sp = al > 20.f ? al : logf(1.f + expf(al));
                gexp[k] = expf(sp * __ldg(L.ssm_a + hv));
                beta[k] = 1.f / (1.f + expf(-__ldcg(&P.s->ab[NVH + hv])));
                }
            }
            if (need_qk) kh_c = kh;
            hv_c = hv;
            float ssq[2] = { 0.f, 0.f };
            if constexpr (XI) {   // A3: the four (sequence, column) chains of this item, reductions interleaved
                float4 qv[2], kv4[2];
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float4 q4 = *(const float4 *)(QN[k] + 4 * lane), k4r = *(const float4 *)(KN[k] + 4 * lane);
                    qv[k]  = make_float4(q4.x * qs[k], q4.y * qs[k], q4.z * qs[k], q4.w * qs[k]);
                    kv4[k] = make_float4(k4r.x * ks[k], k4r.y * ks[k], k4r.z * ks[k], k4r.w * ks[k]);
                }
                float r[2][2];
#pragma unroll
                for (int k = 0; k < 2; ++k)
#pragma unroll
                    for (int j = 0; j < 2; ++j) { const float4 a = st[k][j]; r[k][j] = a.x * kv4[k].x + a.y * kv4[k].y + a.z * kv4[k].z + a.w * kv4[k].w; }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)   // warp_sum of each chain, levels interleaved across chains
#pragma unroll
                    for (int k = 0; k < 2; ++k)
#pragma unroll
                        for (int j = 0; j < 2; ++j) r[k][j] += __shfl_xor_sync(0xffffffffu, r[k][j], o);
#pragma unroll
                for (int k = 0; k < 2; ++k)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        float4 a = st[k][j];
                        const float delta = (VV[k][warp * 2 + j] - gexp[k] * r[k][j]) * beta[k];
                        a.x = gexp[k] * a.x + kv4[k].x * delta; a.y = gexp[k] * a.y + kv4[k].y * delta;
                        a.z = gexp[k] * a.z + kv4[k].z * delta; a.w = gexp[k] * a.w + kv4[k].w * delta;
                        st[k][j] = a;
                        r[k][j] = a.x * qv[k].x + a.y * qv[k].y + a.z * qv[k].z + a.w * qv[k].w;
                    }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int k = 0; k < 2; ++k)
#pragma unroll
                        for (int j = 0; j < 2; ++j) r[k][j] += __shfl_xor_sync(0xffffffffu, r[k][j], o);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const Params & P = PP.p[k];
                    float4 * so = (float4 *)(P.S_out + ((size_t)hv * SD + col0) * SD) + lane;
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float att = r[k][j] * rsqrtf((float)SD);
                        so[j * SD / 4] = st[k][j];
                        if (lane == 0) P.s->attn[hv * SD + col0 + j] = att;
                        ssq[k] += att * att;
                    }
                }
            } else {
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const Params & P = PP.p[k];
                const float4 q4 = *(const float4 *)(QN[k] + 4 * lane), k4r = *(const float4 *)(KN[k] + 4 * lane);
                const float4 qv = make_float4(q4.x * qs[k], q4.y * qs[k], q4.z * qs[k], q4.w * qs[k]);
                const float4 kv4 = make_float4(k4r.x * ks[k], k4r.y * ks[k], k4r.z * ks[k], k4r.w * ks[k]);
                float4 * so = (float4 *)(P.S_out + ((size_t)hv * SD + col0) * SD) + lane;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float4 a = st[k][j];
                    float kvd = warp_sum(a.x * kv4.x + a.y * kv4.y + a.z * kv4.z + a.w * kv4.w);
                    float delta = (VV[k][warp * 2 + j] - gexp[k] * kvd) * beta[k];
                    a.x = gexp[k] * a.x + kv4.x * delta; a.y = gexp[k] * a.y + kv4.y * delta;
                    a.z = gexp[k] * a.z + kv4.z * delta; a.w = gexp[k] * a.w + kv4.w * delta;
                    float att = warp_sum(a.x * qv.x + a.y * qv.y + a.z * qv.z + a.w * qv.w) * rsqrtf((float)SD);
                    so[j * SD / 4] = a;
                    if (lane == 0) P.s->attn[hv * SD + col0 + j] = att;
                    ssq[k] += att * att;
                }
            }
            }
            if constexpr (XI) {   // A3: readers of qn/kn/vv/red of this item are all before this barrier
                if (lane == 0) { sq2[par][0][warp] = ssq[0]; sq2[par][1][warp] = ssq[1]; }
                __syncthreads();
                if (threadIdx.x == 0 || threadIdx.x == 32) {
                    const int k = threadIdx.x >> 5;
                    const float * rk = sq2[par][k];
                    float t = 0.f; for (int i = 0; i < NW; ++i) t += rk[i];
                    (k ? PP.p[1].s : PP.p[0].s)->ssq[hv * NCG + cg] = t; (k ? hc1 : hc)[hv]++;
                }
                par ^= 1;
                continue;
            }
            __syncthreads();
            if (lane == 0) { red[warp] = ssq[0]; red1[warp] = ssq[1]; }
            __syncthreads();
            if (threadIdx.x == 0 || threadIdx.x == 32) {
                const int k = threadIdx.x >> 5;
                const float * rk = k ? red1 : red;   // selects, not a dynamically indexed pointer array (no local memory)
                float t = 0.f; for (int i = 0; i < NW; ++i) t += rk[i];
                (k ? PP.p[1].s : PP.p[0].s)->ssq[hv * NCG + cg] = t; (k ? hc1 : hc)[hv]++;
            }
            __syncthreads();
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x < 2 * NVH) {
            const int k = threadIdx.x / NVH, h = threadIdx.x % NVH;
            unsigned * const hk = k ? hc1 : hc, * const fk = k ? fin1 : fin;
            if (hk[h] != 0) {
                const unsigned c = hk[h];
                const unsigned old = atomicAdd(&(k ? PP.p[1].s : PP.p[0].s)->hcnt[h], c);
                if ((old % NCG) + c == NCG) { fk[atomicAdd(&fk[NVH], 1u)] = h; }
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const Params & P = PP.p[k];
            Scratch * s = P.s;
            const unsigned nf = FN[k][NVH];
            if (nf) {
                __threadfence();
                for (unsigned f = threadIdx.x >> 7; f < nf; f += NT / SD) {
                    const int hv = FN[k][f], t = threadIdx.x & (SD - 1), i = hv * SD + t;
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
    }
    grid_barrier(s0);
    {   // ---- phase C ----
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

} // namespace occamy_gdn
