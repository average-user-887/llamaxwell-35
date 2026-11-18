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

