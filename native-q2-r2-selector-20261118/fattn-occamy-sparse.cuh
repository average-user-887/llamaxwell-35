#pragma once

// Approximate attention: omit low-proxy prefix blocks and preserve the recent causal tail.
constexpr int OFS_BLOCK = 256;
constexpr int OFS_KEEP = 16;
constexpr int OFS_RECENT = 4096;
constexpr int OFS_MAX_BLOCKS = 512;

struct ofa_sparse_plan {
    int ok, invalid, n_blocks, recent_begin, visible_end, units, kept, omitted, fallback;
    int sparse_warps, full_warps;
    int blocks[OFS_KEEP];
};

template <int NQ>
__global__ void __launch_bounds__(128) ofa_sparse_init(ofa_params p, ofa_sparse_plan * plans) {
    __shared__ int ends[4], invalids[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int end = 0, invalid = 0;
    {
        const int r = blockIdx.x/p.nkvh;
        const half * mask = (const half *) (p.mask + (size_t) r*p.nbm1);
        for (int t = threadIdx.x; t < p.n_kv; t += blockDim.x) {
            const float m = __half2float(mask[t]);
            if (isfinite(m)) end = max(end, t + 1);
            else if (m != -INFINITY) invalid = 1;
        }
    }
#pragma unroll
    for (int d = 16; d > 0; d /= 2) {
        end = max(end, __shfl_xor_sync(0xFFFFFFFF, end, d, 32));
        invalid |= __shfl_xor_sync(0xFFFFFFFF, invalid, d, 32);
    }
    if (lane == 0) { ends[warp] = end; invalids[warp] = invalid; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 0; w < 4; ++w) { end = max(end, ends[w]); invalid |= invalids[w]; }
        ofa_sparse_plan & plan = plans[blockIdx.x];
        plan.ok = 0; plan.invalid = invalid; plan.n_blocks = 0; plan.fallback = 0;
        plan.sparse_warps = 0; plan.full_warps = 0;
        plan.visible_end = end;
        plan.recent_begin = max(0, end - OFS_RECENT)/OFS_BLOCK*OFS_BLOCK;
        plan.units = 0; plan.kept = 0; plan.omitted = end;
    }
}

template <int NQ>
__global__ void __launch_bounds__(128) ofa_sparse_probe(ofa_params p, ofa_sparse_plan * plans, float * scores, int n_blocks) {
    __shared__ float bests[4];
    const int b = blockIdx.x, plan_index = blockIdx.y;
    const int kvh = plan_index % p.nkvh, row = plan_index/p.nkvh;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    ofa_sparse_plan & plan = plans[plan_index];
    if (b >= plan.recent_begin/OFS_BLOCK) {
        if (threadIdx.x == 0) scores[(size_t) plan_index*n_blocks + b] = -INFINITY;
        return;
    }
    const int t = b*OFS_BLOCK + warp*(OFS_BLOCK/4);
    float k[8];
    ofa_cvt8(__ldg((const uint4 *) (p.K + (size_t) kvh*p.nbk2 + (size_t) t*p.nbk1 + lane*16)), k);
    float best = -INFINITY;
    {
        const int r = row;
        const float mask = __half2float(((const half *) (p.mask + (size_t) r*p.nbm1))[t]);
#pragma unroll
        for (int h = 0; h < OFA_G; ++h) {
            const float4 * q = (const float4 *) (p.Q + (size_t) r*p.nbq1 + (size_t) (kvh*OFA_G + h)*p.nbq2) + 2*lane;
            const float4 a = q[0], z = q[1];
            const float v[8] = {a.x, a.y, a.z, a.w, z.x, z.y, z.z, z.w};
            float acc = v[0]*k[0];
#pragma unroll
            for (int d = 1; d < 8; ++d) acc = fmaf(v[d], k[d], acc);
#pragma unroll
            for (int d = 16; d > 0; d /= 2) acc += __shfl_xor_sync(0xFFFFFFFF, acc, d, 32);
            const float score = acc*p.scale_log2 + mask*1.4426950408889634f;
            if (!isfinite(score) && score != -INFINITY && lane == 0) atomicExch(&plan.invalid, 1);
            best = fmaxf(best, score);
        }
    }
    if (lane == 0) bests[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 0; w < 4; ++w) best = fmaxf(best, bests[w]);
        scores[(size_t) plan_index*n_blocks + b] = best;
    }
}

// Score-invalid bits are applied only to old blocks after the visible end is known.
// This preserves the original exclusion of recent blocks from proxy validity checks.
struct ofa_sparse_meta { int end, mask_invalid, score_invalid; };

template <int NQ>
__global__ void __launch_bounds__(128) ofa_sparse_scan_probe_r2(ofa_params p, ofa_sparse_meta * metas, float * scores, int n_blocks) {
    __shared__ float bests[4];
    __shared__ int ends[4], mask_bad[4], score_bad[4];
    const int b = blockIdx.x, plan_index = blockIdx.y;
    const int kvh = plan_index % p.nkvh, row = plan_index/p.nkvh;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // Each CTA scans its own 256-key mask block, distributed over all old/cache blocks.
    const half * row_mask = (const half *) (p.mask + (size_t) row*p.nbm1);
    int end = 0, invalid_mask = 0, invalid_score = 0;
    for (int t = b*OFS_BLOCK + threadIdx.x; t < min((b + 1)*OFS_BLOCK, p.n_kv); t += blockDim.x) {
        const float m = __half2float(row_mask[t]);
        if (isfinite(m)) end = max(end, t + 1);
        else if (m != -INFINITY) invalid_mask = 1;
    }
#pragma unroll
    for (int d = 16; d > 0; d /= 2) {
        end = max(end, __shfl_xor_sync(0xFFFFFFFF, end, d, 32));
        invalid_mask |= __shfl_xor_sync(0xFFFFFFFF, invalid_mask, d, 32);
    }
    if (lane == 0) { ends[warp] = end; mask_bad[warp] = invalid_mask; }
    const int t = b*OFS_BLOCK + warp*(OFS_BLOCK/4);
    float best = -INFINITY;
    if (t < p.n_kv) {
    float k[8];
    ofa_cvt8(__ldg((const uint4 *) (p.K + (size_t) kvh*p.nbk2 + (size_t) t*p.nbk1 + lane*16)), k);
    {
        const int r = row;
        const float mask = __half2float(((const half *) (p.mask + (size_t) r*p.nbm1))[t]);
#pragma unroll
        for (int h = 0; h < OFA_G; ++h) {
            const float4 * q = (const float4 *) (p.Q + (size_t) r*p.nbq1 + (size_t) (kvh*OFA_G + h)*p.nbq2) + 2*lane;
            const float4 a = __ldg(q), z = __ldg(q + 1);
            const float v[8] = {a.x, a.y, a.z, a.w, z.x, z.y, z.z, z.w};
            float acc = v[0]*k[0];
#pragma unroll
            for (int d = 1; d < 8; ++d) acc = fmaf(v[d], k[d], acc);
#pragma unroll
            for (int d = 16; d > 0; d /= 2) acc += __shfl_xor_sync(0xFFFFFFFF, acc, d, 32);
            const float score = acc*p.scale_log2 + mask*1.4426950408889634f;
            if (!isfinite(score) && score != -INFINITY && lane == 0) invalid_score = 1;
            best = fmaxf(best, score);
        }
    }
    }
    if (lane == 0) { bests[warp] = best; score_bad[warp] = invalid_score; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int block_end = 0, block_mask_bad = 0, block_score_bad = 0;
        for (int w = 0; w < 4; ++w) {
            best = fmaxf(best, bests[w]);
            block_end = max(block_end, ends[w]);
            block_mask_bad |= mask_bad[w]; block_score_bad |= score_bad[w];
        }
        const size_t index = (size_t) plan_index*n_blocks + b;
        scores[index] = best;
        metas[index] = {block_end, block_mask_bad, block_score_bad};
    }
}

static __device__ __forceinline__ bool ofa_sparse_plan_valid(const ofa_sparse_plan & p, int n_kv) {
    if (!p.ok || p.invalid || p.n_blocks < 0 || p.n_blocks > OFS_KEEP) return false;
    if (p.recent_begin < 0 || p.recent_begin % OFS_BLOCK || p.visible_end <= p.recent_begin || p.visible_end > n_kv) return false;
    int prev = -1;
    for (int j = 0; j < p.n_blocks; ++j) {
        const int b = p.blocks[j];
        if (b <= prev || b < 0 || b >= p.recent_begin/OFS_BLOCK) return false;
        prev = b;
    }
    const int kept = p.n_blocks*OFS_BLOCK + p.visible_end - p.recent_begin;
    return p.kept == kept && p.omitted == p.visible_end - kept && p.units == p.n_blocks*(OFS_BLOCK/OFA_T) + (p.visible_end - p.recent_begin + OFA_T - 1)/OFA_T;
}

__global__ void __launch_bounds__(128) ofa_sparse_select(ofa_sparse_plan * plans, const float * scores, int n_blocks) {
    __shared__ float work[OFS_MAX_BLOCKS], warp_value[4];
    __shared__ int warp_index[4];
    const int kvh = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    ofa_sparse_plan & plan = plans[kvh];
    const int n_old = plan.recent_begin/OFS_BLOCK;
    for (int j = threadIdx.x; j < n_blocks; j += blockDim.x) work[j] = scores[(size_t) kvh*n_blocks + j];
    __syncthreads();
    for (int pick = 0; pick < OFS_KEEP; ++pick) {
        float value = -INFINITY;
        int index = INT32_MAX;
        for (int j = threadIdx.x; j < n_old; j += blockDim.x) {
            const float v = work[j];
            if (v > value || (v == value && j < index)) { value = v; index = j; }
        }
#pragma unroll
        for (int d = 16; d > 0; d /= 2) {
            const float v = __shfl_xor_sync(0xFFFFFFFF, value, d, 32);
            const int i = __shfl_xor_sync(0xFFFFFFFF, index, d, 32);
            if (v > value || (v == value && i < index)) { value = v; index = i; }
        }
        if (lane == 0) { warp_value[warp] = value; warp_index[warp] = index; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < 4; ++w) {
                const float v = warp_value[w]; const int i = warp_index[w];
                if (v > value || (v == value && i < index)) { value = v; index = i; }
            }
            if (isfinite(value) && index >= 0 && index < n_old && plan.n_blocks < OFS_KEEP) {
                plan.blocks[plan.n_blocks++] = index;
                work[index] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int j = 1; j < plan.n_blocks; ++j) {
            const int b = plan.blocks[j]; int i = j;
            while (i > 0 && plan.blocks[i - 1] > b) { plan.blocks[i] = plan.blocks[i - 1]; --i; }
            plan.blocks[i] = b;
        }
        plan.kept = plan.n_blocks*OFS_BLOCK + plan.visible_end - plan.recent_begin;
        plan.omitted = plan.visible_end - plan.kept;
        plan.units = plan.n_blocks*(OFS_BLOCK/OFA_T) + (plan.visible_end - plan.recent_begin + OFA_T - 1)/OFA_T;
        plan.ok = !plan.invalid && plan.visible_end > 0 && (n_old == 0 || plan.n_blocks > 0);
    }
}

__global__ void __launch_bounds__(128) ofa_sparse_select_r2(ofa_sparse_plan * plans, const float * scores, const ofa_sparse_meta * metas, int n_blocks) {
    __shared__ float work[OFS_MAX_BLOCKS], warp_value[4];
    __shared__ int warp_index[4];
    const int kvh = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    ofa_sparse_plan & plan = plans[kvh];
    __shared__ int ends[4], invalids[4];
    int end = 0, invalid = 0;
    for (int j = threadIdx.x; j < n_blocks; j += blockDim.x) {
        const ofa_sparse_meta m = metas[(size_t) kvh*n_blocks + j];
        end = max(end, m.end); invalid |= m.mask_invalid;
    }
#pragma unroll
    for (int d = 16; d > 0; d /= 2) {
        end = max(end, __shfl_xor_sync(0xFFFFFFFF, end, d, 32));
        invalid |= __shfl_xor_sync(0xFFFFFFFF, invalid, d, 32);
    }
    if (lane == 0) { ends[warp] = end; invalids[warp] = invalid; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 0; w < 4; ++w) { end = max(end, ends[w]); invalid |= invalids[w]; }
        plan.ok = 0; plan.invalid = invalid; plan.n_blocks = 0; plan.fallback = 0;
        plan.sparse_warps = 0; plan.full_warps = 0;
        plan.visible_end = end;
        plan.recent_begin = max(0, end - OFS_RECENT)/OFS_BLOCK*OFS_BLOCK;
        plan.units = 0; plan.kept = 0; plan.omitted = end;
    }
    __syncthreads();
    const int n_old = plan.recent_begin/OFS_BLOCK;
    invalid = 0;
    for (int j = threadIdx.x; j < n_old; j += blockDim.x) invalid |= metas[(size_t) kvh*n_blocks + j].score_invalid;
#pragma unroll
    for (int d = 16; d > 0; d /= 2) invalid |= __shfl_xor_sync(0xFFFFFFFF, invalid, d, 32);
    if (lane == 0) invalids[warp] = invalid;
    __syncthreads();
    if (threadIdx.x == 0) for (int w = 0; w < 4; ++w) plan.invalid |= invalids[w];
    for (int j = threadIdx.x; j < n_blocks; j += blockDim.x) work[j] = scores[(size_t) kvh*n_blocks + j];
    __syncthreads();
    for (int pick = 0; pick < OFS_KEEP; ++pick) {
        float value = -INFINITY;
        int index = INT32_MAX;
        for (int j = threadIdx.x; j < n_old; j += blockDim.x) {
            const float v = work[j];
            if (v > value || (v == value && j < index)) { value = v; index = j; }
        }
#pragma unroll
        for (int d = 16; d > 0; d /= 2) {
            const float v = __shfl_xor_sync(0xFFFFFFFF, value, d, 32);
            const int i = __shfl_xor_sync(0xFFFFFFFF, index, d, 32);
            if (v > value || (v == value && i < index)) { value = v; index = i; }
        }
        if (lane == 0) { warp_value[warp] = value; warp_index[warp] = index; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < 4; ++w) {
                const float v = warp_value[w]; const int i = warp_index[w];
                if (v > value || (v == value && i < index)) { value = v; index = i; }
            }
            if (isfinite(value) && index >= 0 && index < n_old && plan.n_blocks < OFS_KEEP) {
                plan.blocks[plan.n_blocks++] = index;
                work[index] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int j = 1; j < plan.n_blocks; ++j) {
            const int b = plan.blocks[j]; int i = j;
            while (i > 0 && plan.blocks[i - 1] > b) { plan.blocks[i] = plan.blocks[i - 1]; --i; }
            plan.blocks[i] = b;
        }
        plan.kept = plan.n_blocks*OFS_BLOCK + plan.visible_end - plan.recent_begin;
        plan.omitted = plan.visible_end - plan.kept;
        plan.units = plan.n_blocks*(OFS_BLOCK/OFA_T) + (plan.visible_end - plan.recent_begin + OFA_T - 1)/OFA_T;
        plan.ok = !plan.invalid && plan.visible_end > 0 && (n_old == 0 || plan.n_blocks > 0);
    }
}

// Encode the reference order: score descending, then smaller block ID first.
// Canonicalize signed zero (equal in the reference float comparisons).
static __device__ __forceinline__ unsigned long long ofa_r2_sort_key(float value, int index) {
    if (isnan(value) || value == -INFINITY) return 0;
    const unsigned bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    const unsigned ordered = bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
    return ((unsigned long long) ordered << 32) | (0xFFFFFFFFu - index);
}

// Each warp ranks 32 contiguous blocks, keeping only the sixteen that
// can reach the global top sixteen. A four-stage merge retains the same bound.
__global__ void __launch_bounds__(512) ofa_sparse_select_r2_top16(
        ofa_sparse_plan * plans, const float * scores, const ofa_sparse_meta * metas, int n_blocks) {
    static_assert(OFS_KEEP == 16 && OFS_MAX_BLOCKS == 512);
    __shared__ int ends[16];
    __shared__ unsigned long long lists[16 * 16], merge[8 * 16];
    const int index = threadIdx.x, lane = index & 31, warp = index >> 5;
    const size_t base = (size_t) blockIdx.x * n_blocks;
    ofa_sparse_meta meta = {};
    if (index < n_blocks) meta = metas[base + index];
    int end = meta.end;
#pragma unroll
    for (int stride = 16; stride > 0; stride /= 2) {
        end = max(end, __shfl_xor_sync(0xFFFFFFFFu, end, stride));
    }
    if (lane == 0) ends[warp] = end;
    __syncthreads();
    end = ends[lane & 15];
#pragma unroll
    for (int stride = 8; stride > 0; stride /= 2) {
        end = max(end, __shfl_xor_sync(0xFFFFFFFFu, end, stride));
    }
    const int recent_begin = max(0, end - OFS_RECENT)/OFS_BLOCK*OFS_BLOCK;
    const int n_old = recent_begin / OFS_BLOCK;
    const int invalid = __syncthreads_or(meta.mask_invalid || (index < n_old && meta.score_invalid));

    unsigned long long key = 0;
    if (n_old > 0) {
        if (warp * 32 < n_old) {
            if (index < n_old) key = ofa_r2_sort_key(scores[base + index], index);
#pragma unroll
            for (int width = 2; width <= 32; width *= 2) {
#pragma unroll
                for (int stride = width / 2; stride > 0; stride /= 2) {
                    const auto other = __shfl_xor_sync(0xFFFFFFFFu, key, stride);
                    const bool high = ((lane & width) == 0) == ((lane & stride) == 0);
                    key = (key < other) == high ? other : key;
                }
            }
        }
        if (lane < 16) lists[warp * 16 + lane] = key;
        __syncthreads();
#pragma unroll
        for (int stage = 0; stage < 4; ++stage) {
            if (warp < (8 >> stage)) {
                const auto * source = (stage & 1) ? merge : lists;
                auto * target = (stage & 1) ? lists : merge;
                // Pair opposite ends of the lists, then bitonic-merge the better half.
                key = source[warp * 32 + (lane < 16 ? lane : 47 - lane)];
                const auto other = __shfl_xor_sync(0xFFFFFFFFu, key, 16);
                key = key < other ? other : key;
#pragma unroll
                for (int stride = 8; stride > 0; stride /= 2) {
                    const auto partner = __shfl_xor_sync(0xFFFFFFFFu, key, stride);
                    key = (key < partner) == ((lane & stride) == 0) ? partner : key;
                }
                if (lane < 16 && stage < 3) target[warp * 16 + lane] = key;
            }
            if (stage < 3) __syncthreads();
        }
    }
    if (warp != 0) return;

    // In the reference, a winning +infinity is never removed, so there are
    // zero selected blocks in that case. Its separate metadata guard still applies.
    const auto first = __shfl_sync(0xFFFFFFFFu, key, 0);
    const bool finite_selection = (first >> 32) != 0xFF800000u;
    const bool selected = lane < 16 && key != 0 && finite_selection;
    const int count = __popc(__ballot_sync(0xFFFFFFFFu, selected));
    int block = selected ? (int) (0xFFFFFFFFu - (unsigned) key) : INT32_MAX;
#pragma unroll
    for (int width = 2; width <= 16; width *= 2) {
#pragma unroll
        for (int stride = width / 2; stride > 0; stride /= 2) {
            const int other = __shfl_xor_sync(0xFFFFFFFFu, block, stride, 16);
            const bool low = ((lane & width) == 0) == ((lane & stride) == 0);
            block = (block > other) == low ? other : block;
        }
    }
    ofa_sparse_plan & plan = plans[blockIdx.x];
    if (lane < count) plan.blocks[lane] = block;
    if (lane == 0) {
        plan.invalid = invalid;
        plan.n_blocks = count;
        plan.recent_begin = recent_begin;
        plan.visible_end = end;
        plan.kept = count*OFS_BLOCK + end - recent_begin;
        plan.omitted = end - plan.kept;
        plan.units = count*(OFS_BLOCK/OFA_T) + (end - recent_begin + OFA_T - 1)/OFA_T;
        plan.ok = !invalid && end > 0 && (n_old == 0 || count > 0);
        plan.fallback = 0;
        plan.sparse_warps = 0;
        plan.full_warps = 0;
    }
}

static __device__ __forceinline__ int ofa_sparse_token(const ofa_sparse_plan & p, int unit) {
    const int old_units = p.n_blocks*(OFS_BLOCK/OFA_T);
    return unit < old_units ? p.blocks[unit/(OFS_BLOCK/OFA_T)]*OFS_BLOCK + (unit % (OFS_BLOCK/OFA_T))*OFA_T : p.recent_begin + (unit - old_units)*OFA_T;
}

int ggml_cuda_occamy_fa_sparse_mode() {
    static const int mode = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_SPARSE");
        return s && strcmp(s, "1") == 0 ? 1 : s && strcmp(s, "2") == 0 ? 2 : 0;
    }();
    return mode;
}

bool ggml_cuda_occamy_fa_sparse_applicable(const ggml_tensor * dst) {
    const int mode = ggml_cuda_occamy_fa_sparse_mode();
    if (!mode || ggml_cuda_info().devices[ggml_cuda_get_device()].cc != 520) return false;
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    if ((q->ne[1] != 1 && q->ne[1] != 2) || q->ne[3] != 1) return false;
    if (!mask || mask->ne[1] < q->ne[1] || mask->ne[3] != 1 || mask->nb[0] != sizeof(half)) return false;
    if (k->ne[1] < 8192 || k->ne[1] > OFS_MAX_BLOCKS*OFS_BLOCK || k->ne[2] <= 0 || k->ne[2] > 256) return false;
    if (dst->ne[0] != OFA_D || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] || dst->ne[3] != 1 || (uintptr_t) dst->data % sizeof(float)) return false;
    for (int i = 1; i < 4; ++i) if (dst->nb[i] % sizeof(float)) return false;
    if (mode == 2 && !ggml_is_contiguous(dst)) return false;
    ggml_tensor d1 = *dst, q1 = *q;
    q1.ne[1] = 1; d1.src[0] = &q1;
    return ggml_cuda_occamy_fa_applicable(&d1);
}

template <int NQ, int NU>
__global__ void __launch_bounds__(32*NU*(2*NQ), (NQ == 1 ? 2 : 1)) ofa_sparse_attention(const ofa_params p, ofa_sparse_plan * plans) {
    constexpr int HG = 4, NG = 2*NQ;
    constexpr int NT = 32*NU*NG;
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int i3    = blockIdx.z;
    const int wid   = threadIdx.x >> 5;
    const int ul    = wid % NU;      // unit lane
    const int g     = wid / NU;
    const int lane  = threadIdx.x & 31;
    const int row = g/2, head_group = g%2;

    __shared__ __align__(16) float s_p[NU*NG][OFA_T*4];
    __shared__ __align__(16) float s_a[NU*NG][4];
    __shared__ __align__(16) float s_o[NU][OFA_G][OFA_D];
    __shared__ float s_m[NU][NQ*OFA_G];
    __shared__ float s_l[NU][NQ*OFA_G];

    // Four query slots per warp, mapped to one or two query rows.
    float q[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4 * qp = (const float4 *) (p.Q + i3*p.nbq3 + (size_t) row*p.nbq1 + (size_t)(kvh*OFA_G + HG*head_group + k)*p.nbq2) + 2*lane;
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
    const int rl = row;
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;
    const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;
    const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3 + (size_t) rl*p.nbm1);

    const int plan_index = row*p.nkvh + kvh;
    const ofa_sparse_plan & plan = plans[plan_index];
    int use_sparse = 0;
    if (lane == 0) {
        use_sparse = ofa_sparse_plan_valid(plan, p.n_kv);
        if (use_sparse) atomicAdd(&plans[plan_index].sparse_warps, 1);
        else { atomicExch(&plans[plan_index].fallback, 1); atomicAdd(&plans[plan_index].full_warps, 1); }
    }
    use_sparse = __shfl_sync(0xFFFFFFFF, use_sparse, 0, 32);
    const int nunits = use_sparse ? plan.units : p.n_kv/OFA_T;
    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);

    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    for (int u = u0 + ul; u < u1; u += NU) {
        const int t0 = use_sparse ? ofa_sparse_token(plan, u) : u*OFA_T;
        const char * kp = Kb + (size_t) t0*nbk1;
        const char * vp = Vb + (size_t) t0*nbv1;
        const half * mp = Mb + t0 + jl;
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
        const int slot = row*OFA_G + HG*head_group + k;
        s_m[ul][slot] = m_run; s_l[ul][slot] = l_run;
    }

    const int blk = (i3*p.nkvh + kvh)*p.nsplit + split;
#pragma unroll
    for (int r = 0; r < NQ; ++r) {
        if (r) __syncthreads();   // pass 0 reads of s_o done
        if (r == row) {
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) {
                float4 * dp = (float4 *) &s_o[ul][HG*head_group + hh][8*lane];
                dp[0] = make_float4(o[hh][0], o[hh][1], o[hh][2], o[hh][3]);
                dp[1] = make_float4(o[hh][4], o[hh][5], o[hh][6], o[hh][7]);
            }
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
            p.part_o[((size_t) blk*NQ*OFA_G + slot)*OFA_D + d] = acc;
            if (d == 0) {
                float L = 0.0f;
#pragma unroll
                for (int w = 0; w < NU; ++w) L += s_l[w][slot]*exp2f(s_m[w][slot] - M);
                p.part_ml[(size_t) blk*NQ*OFA_G + slot] = make_float2(M, L);
            }
        }
    }
}

static std::atomic<long long> ofa_sparse_calls[3];

template <int NQ>
static void ofa_sparse_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const int nkvh = (int) k->ne[2], n_kv = (int) k->ne[1];
    const int nplans = NQ*nkvh;
    static const bool revision2 = []() { const char * s = getenv("GGML_CUDA_OCCAMY_FA_SPARSE_R2"); return s && strcmp(s, "1") == 0; }();
    static const bool selector_check = []() { const char * s = getenv("GGML_CUDA_OCCAMY_FA_SPARSE_R2_CHECK"); return s && strcmp(s, "1") == 0; }();
    const bool use_top16 = NQ == 2 && nkvh == 2 && n_kv >= 80128 && n_kv <= 83200 && n_kv % OFS_BLOCK == 0 && [] {
        static const bool enabled = [] {
            const char * s = getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16");
            return s && strcmp(s, "1") == 0;
        }();
        return enabled;
    }();
    constexpr int NU = 4, NG = 2*NQ, NT = 32*NU*NG;
    const int id = ggml_cuda_get_device();
    static const int blocks_per_sm = []() {
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, ofa_sparse_attention<NQ, NU>, NT, 0));
        return std::max(1, n);
    }();
    const int resident = blocks_per_sm*ggml_cuda_info().devices[id].nsm;
    const int max_selected = OFS_KEEP*OFS_BLOCK + OFS_RECENT + OFS_BLOCK;
    const int nsplit = std::max(1, std::min((resident + nkvh - 1)/nkvh, max_selected/(OFA_T*NU*4)));
    const size_t nblk = (size_t) nsplit*nkvh;
    const int n_blocks = (n_kv + OFS_BLOCK - 1)/OFS_BLOCK;
    ggml_cuda_pool_alloc<ofa_sparse_plan> plans(ctx.pool(), nplans);
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), (size_t) nplans*n_blocks);
    ggml_cuda_pool_alloc<ofa_sparse_meta> metas(ctx.pool(), revision2 ? (size_t) nplans*n_blocks : 1);
    ggml_cuda_pool_alloc<float> part_o(ctx.pool(), nblk*NQ*OFA_G*OFA_D);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), nblk*NQ*OFA_G);
    float scale;
    memcpy(&scale, (const float *) dst->op_params, sizeof(scale));
    ofa_params p;
    p.Q = (const char *) q->data; p.K = (const char *) k->data; p.V = (const char *) v->data; p.mask = (const char *) mask->data;
    p.part_o = part_o.get(); p.part_ml = part_ml.get();
    p.n_kv = n_kv; p.nsplit = nsplit; p.nkvh = nkvh; p.mask_ne1 = (int) mask->ne[1]; p.mask_ne3 = 1;
    p.scale_log2 = scale*1.4426950408889634f;
    p.nbq1 = q->nb[1]; p.nbq2 = q->nb[2]; p.nbq3 = q->nb[3];
    p.nbk1 = k->nb[1]; p.nbk2 = k->nb[2]; p.nbk3 = k->nb[3];
    p.nbv1 = v->nb[1]; p.nbv2 = v->nb[2]; p.nbv3 = v->nb[3];
    p.nbm1 = mask->nb[1]; p.nbm3 = mask->nb[3];
    const long long nc = ++ofa_sparse_calls[NQ];
    static const bool profile_enabled = []() {
        const char * s = getenv("GGML_CUDA_OCCAMY_FA_SPARSE_PROFILE");
        return s && strcmp(s, "1") == 0;
    }();
    const bool profile = profile_enabled && (nc <= 10 || nc % 100 == 0);
    cudaStream_t stream = ctx.stream();
    cudaEvent_t events[6];
    if (profile) {
        for (int i = 0; i < 6; ++i) CUDA_CHECK(cudaEventCreate(&events[i]));
        CUDA_CHECK(cudaEventRecord(events[0], stream));
    }
    if (revision2) {
        if (profile) CUDA_CHECK(cudaEventRecord(events[4], stream));
        ofa_sparse_scan_probe_r2<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, metas.get(), scores.get(), n_blocks);
        CUDA_CHECK(cudaGetLastError());
        if (profile) CUDA_CHECK(cudaEventRecord(events[5], stream));
        if (use_top16) {
            ofa_sparse_select_r2_top16<<<nplans, 512, 0, stream>>>(plans.get(), scores.get(), metas.get(), n_blocks);
            if (nc == 1) fprintf(stderr, "q2_r2_top16_dispatch n_q=2 n_kv=%d nkvh=%d plans=%d\n", n_kv, nkvh, nplans);
        } else {
            ofa_sparse_select_r2<<<nplans, 128, 0, stream>>>(plans.get(), scores.get(), metas.get(), n_blocks);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        ofa_sparse_init<NQ><<<nplans, 128, 0, stream>>>(p, plans.get());
        CUDA_CHECK(cudaGetLastError());
        if (profile) CUDA_CHECK(cudaEventRecord(events[4], stream));
        ofa_sparse_probe<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, plans.get(), scores.get(), n_blocks);
        CUDA_CHECK(cudaGetLastError());
        if (profile) CUDA_CHECK(cudaEventRecord(events[5], stream));
        ofa_sparse_select<<<nplans, 128, 0, stream>>>(plans.get(), scores.get(), n_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
    if (profile) CUDA_CHECK(cudaEventRecord(events[1], stream));
    ofa_sparse_attention<NQ, NU><<<dim3(nsplit, nkvh, 1), NT, 0, stream>>>(p, plans.get());
    CUDA_CHECK(cudaGetLastError());
    if (profile) CUDA_CHECK(cudaEventRecord(events[2], stream));
    ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, NQ, 1), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), (char *) dst->data, nsplit, nkvh, NQ, dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
    if (profile) CUDA_CHECK(cudaEventRecord(events[3], stream));
    if constexpr (NQ == 2) {
        static const bool check_top16 = [] {
            const char * s = getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16_CHECK");
            return s && strcmp(s, "1") == 0;
        }();
        if (revision2 && use_top16 && check_top16 && nc == 1) {
            ggml_cuda_pool_alloc<ofa_sparse_plan> old_plans(ctx.pool(), nplans);
            ggml_cuda_pool_alloc<ofa_sparse_meta> old_metas(ctx.pool(), (size_t) nplans*n_blocks);
            ggml_cuda_pool_alloc<float> old_scores(ctx.pool(), (size_t) nplans*n_blocks);
            ggml_cuda_pool_alloc<float> old_output(ctx.pool(), ggml_nelements(dst));
            GGML_ASSERT(ggml_is_contiguous(dst));
            ofa_sparse_scan_probe_r2<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, old_metas.get(), old_scores.get(), n_blocks);
            ofa_sparse_select_r2<<<nplans, 128, 0, stream>>>(old_plans.get(), old_scores.get(), old_metas.get(), n_blocks);
            ofa_sparse_attention<NQ, NU><<<dim3(nsplit, nkvh, 1), NT, 0, stream>>>(p, old_plans.get());
            ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, NQ, 1), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), (char *) old_output.get(), nsplit, nkvh, NQ, dst->nb[1], dst->nb[2], dst->nb[3]);
            CUDA_CHECK(cudaGetLastError());
            std::vector<ofa_sparse_plan> a(nplans), b(nplans);
            std::vector<float> sa((size_t) nplans*n_blocks), sb(sa.size()), va(ggml_nelements(dst)), vb(va.size());
            auto copy = [&](auto & output, const void * device) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), device, output.size()*sizeof(output[0]), cudaMemcpyDeviceToHost, stream));
            };
            copy(a, plans.get()); copy(b, old_plans.get());
            copy(sa, scores.get()); copy(sb, old_scores.get());
            copy(va, dst->data); copy(vb, old_output.get());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int plan_diff = 0, score_diff = 0, output_diff = 0, nonfinite = 0;
            for (int i = 0; i < nplans; ++i) {
                const size_t bytes = offsetof(ofa_sparse_plan, blocks) + b[i].n_blocks*sizeof(int);
                plan_diff += memcmp(&a[i], &b[i], bytes) != 0;
            }
            for (size_t i = 0; i < sa.size(); ++i) score_diff += memcmp(&sa[i], &sb[i], sizeof(float)) != 0;
            for (size_t i = 0; i < va.size(); ++i) {
                output_diff += memcmp(&va[i], &vb[i], sizeof(float)) != 0;
                nonfinite += !std::isfinite(va[i]) || !std::isfinite(vb[i]);
            }
            fprintf(stderr, "q2_r2_top16_check n_kv=%d plans=%d scores=%zu outputs=%zu plan_diff=%d score_diff=%d output_diff=%d nonfinite=%d\n",
                    n_kv, nplans, sa.size(), va.size(), plan_diff, score_diff, output_diff, nonfinite);
            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && output_diff == 0 && nonfinite == 0);
        }
    }
    if (nc == 1 || profile || (revision2 && selector_check)) {
        std::vector<ofa_sparse_plan> host_plans(nplans);
        CUDA_CHECK(cudaMemcpyAsync(host_plans.data(), plans.get(), nplans*sizeof(ofa_sparse_plan), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int h = 0; h < nplans; ++h) {
            const ofa_sparse_plan & s = host_plans[h];
            const bool active = s.ok && !s.fallback;
            fprintf(stderr, "occamy_sparse_plan call=%lld n_q=%d row=%d kvh=%d n_kv=%d ok=%d invalid=%d fallback=%d engaged_warps=%d full_warps=%d expected_warps=%d selected=%d omitted=%d padded=%d recent_begin=%d visible_end=%d old_blocks=%d units=%d block_ids=", nc, NQ, h/nkvh, h%nkvh, n_kv, s.ok, s.invalid, s.fallback, s.sparse_warps, s.full_warps, nsplit*2*NU, active ? s.kept : n_kv, active ? s.omitted : 0, n_kv - s.visible_end, s.recent_begin, s.visible_end, s.n_blocks, s.units);
            for (int j = 0; j < s.n_blocks && j < OFS_KEEP; ++j) fprintf(stderr, "%s%d", j ? "," : "", s.blocks[j]);
            fprintf(stderr, "\n");
        }
        if (revision2 && selector_check) {
            ggml_cuda_pool_alloc<ofa_sparse_plan> reference(ctx.pool(), nplans);
            ggml_cuda_pool_alloc<float> reference_scores(ctx.pool(), (size_t) nplans*n_blocks);
            ofa_sparse_init<NQ><<<nplans, 128, 0, stream>>>(p, reference.get());
            ofa_sparse_probe<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, reference.get(), reference_scores.get(), n_blocks);
            ofa_sparse_select<<<nplans, 128, 0, stream>>>(reference.get(), reference_scores.get(), n_blocks);
            CUDA_CHECK(cudaGetLastError());
            std::vector<ofa_sparse_plan> reference_plans(nplans);
            std::vector<float> reference_values((size_t) nplans*n_blocks), candidate_values((size_t) nplans*n_blocks);
            CUDA_CHECK(cudaMemcpyAsync(reference_plans.data(), reference.get(), nplans*sizeof(ofa_sparse_plan), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(reference_values.data(), reference_scores.get(), reference_values.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(candidate_values.data(), scores.get(), candidate_values.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int plan_diff = 0, score_diff = 0;
            for (int h = 0; h < nplans; ++h) {
                const ofa_sparse_plan & a = host_plans[h], & b = reference_plans[h];
                bool mismatch = a.ok != b.ok || a.invalid != b.invalid || a.n_blocks != b.n_blocks || a.recent_begin != b.recent_begin || a.visible_end != b.visible_end || a.units != b.units || a.kept != b.kept || a.omitted != b.omitted;
                for (int j = 0; j < std::min(OFS_KEEP, b.n_blocks); ++j) mismatch |= a.blocks[j] != b.blocks[j];
                plan_diff += mismatch;
                for (int j = 0; j < b.recent_begin/OFS_BLOCK; ++j) {
                    const size_t i = (size_t) h*n_blocks + j;
                    score_diff += memcmp(&reference_values[i], &candidate_values[i], sizeof(float)) != 0;
                }
            }
            fprintf(stderr, "occamy_sparse_r2_check call=%lld n_q=%d n_kv=%d plans=%d plan_diff=%d old_score_bit_diff=%d\n", nc, NQ, n_kv, nplans, plan_diff, score_diff);
            GGML_ASSERT(plan_diff == 0 && score_diff == 0);
        }
        if (profile) {
            float selector = 0, attention = 0, combine = 0, scan = 0, probe = 0, topk = 0;
            if (!revision2) CUDA_CHECK(cudaEventElapsedTime(&scan, events[0], events[4]));
            CUDA_CHECK(cudaEventElapsedTime(&probe, events[4], events[5]));
            CUDA_CHECK(cudaEventElapsedTime(&topk, events[5], events[1]));
            CUDA_CHECK(cudaEventElapsedTime(&selector, events[0], events[1]));
            CUDA_CHECK(cudaEventElapsedTime(&attention, events[1], events[2]));
            CUDA_CHECK(cudaEventElapsedTime(&combine, events[2], events[3]));
            fprintf(stderr, "occamy_sparse_timing call=%lld n_q=%d selector_ms=%.6f attention_ms=%.6f combine_ms=%.6f total_ms=%.6f\n", nc, NQ, selector, attention, combine, selector + attention + combine);
            fprintf(stderr, "occamy_sparse_selector_timing call=%lld n_q=%d scan_ms=%.6f probe_ms=%.6f topk_ms=%.6f revision=%d\n", nc, NQ, scan, probe, topk, revision2 ? 2 : 1);
            for (int i = 0; i < 6; ++i) CUDA_CHECK(cudaEventDestroy(events[i]));
        }
    }
    if (nc == 1 || nc == 10 || nc == 100 || nc % 1000 == 0) {
        fprintf(stderr, "ggml_cuda: occamy_sparse_probe dispatch calls=%lld n_q=%d n_kv=%d nkvh=%d nsplit=%d blocks_per_sm=%d approximate=1 keep_old_blocks=%d recent=%d revision=%d\n", nc, NQ, n_kv, nkvh, nsplit, blocks_per_sm, OFS_KEEP, OFS_RECENT, revision2 ? 2 : 1);
    }
}

void ggml_cuda_occamy_fa_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[0]->ne[1] == 1) ofa_sparse_launch<1>(ctx, dst);
    else ofa_sparse_launch<2>(ctx, dst);
}

__global__ void ofa_sparse_compare_kernel(const float * a, const float * b, int n, unsigned int * result) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        const float x = a[i], y = b[i];
        if (!isfinite(x) || !isfinite(y)) { atomicAdd(result + 3, 1u); continue; }
        const float e = fabsf(x - y);
        atomicMax(result, __float_as_uint(e));
        atomicMax(result + 1, __float_as_uint(fabsf(y)));
        atomicMax(result + 2, __float_as_uint(e/fmaxf(fabsf(y), 1e-2f)));
    }
}

void ggml_cuda_occamy_fa_sparse_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n = (int) ggml_nelements(dst);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);
    ggml_tensor candidate = *dst;
    candidate.data = tmp.get();
    ggml_cuda_occamy_fa_sparse(ctx, &candidate);
    ggml_cuda_pool_alloc<unsigned int> res(ctx.pool(), 4);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(res.get(), 0, 4*sizeof(unsigned int), stream));
    ofa_sparse_compare_kernel<<<16, 256, 0, stream>>>(tmp.get(), (const float *) dst->data, n, res.get());
    CUDA_CHECK(cudaGetLastError());
    unsigned int values[4];
    CUDA_CHECK(cudaMemcpyAsync(values, res.get(), sizeof(values), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float diffs[3];
    memcpy(diffs, values, sizeof(diffs));
    fprintf(stderr, "occamy_sparse_approx_error name=%s n_q=%lld n_kv=%lld max_abs=%.6e max_full=%.6e max_rel=%.6e nonfinite=%u exactness_expected=0 stock_retained=1\n", dst->name, (long long) dst->src[0]->ne[1], (long long) dst->src[1]->ne[1], diffs[0], diffs[1], diffs[2], values[3]);
}
