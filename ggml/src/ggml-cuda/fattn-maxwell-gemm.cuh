#pragma once

#include "common.cuh"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>


// occamy_fa_gemm counters: one reason per guard clause; printed at first use and at
// CUDA backend free / process exit (only when GGML_CUDA_FA_MAXWELL_GEMM=1).
enum occamy_fa_gemm_reason {
    OFG_cc, OFG_nullsrc, OFG_qtype, OFG_ktype, OFG_vtype, OFG_dst, OFG_sinks, OFG_headdim, OFG_heads, OFG_ne3,
    OFG_q_lt_64, OFG_q_gt_1024, OFG_keys_lt_4096, OFG_keys_gt_98304, OFG_shape, OFG_stride, OFG_mask, OFG_opparams,
    OFG_COUNT
};
static const char * const occamy_fa_gemm_reason_names[OFG_COUNT] = {
    "cc", "nullsrc", "qtype", "ktype", "vtype", "dst", "sinks", "headdim", "heads", "ne3",
    "q_lt_64", "q_gt_1024", "keys_lt_4096", "keys_gt_98304", "shape", "stride", "mask", "opparams",
};
static std::atomic<uint64_t> occamy_fa_gemm_engaged{0};
static std::atomic<uint64_t> occamy_fa_gemm_fallbacks[OFG_COUNT];
static std::atomic<bool> occamy_fa_gemm_used{false};
// WP2 fused softmax (GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1): calls = engaged FA ops on the fused path,
// softmax_launches = fused split-softmax kernel launches. Line printed only when the flag is on.
static std::atomic<uint64_t> occamy_fa_gemm_fused_calls{0};
static std::atomic<uint64_t> occamy_fa_gemm_fused_launches{0};

static bool occamy_fa_gemm_fused_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_FA_MAXWELL_GEMM_FUSED");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static void occamy_fa_gemm_print(const char * when) {
    uint64_t total = 0;
    char reasons[1024];
    size_t off = 0;
    reasons[0] = '\0';
    for (int i = 0; i < OFG_COUNT; ++i) {
        const uint64_t n = occamy_fa_gemm_fallbacks[i].load(std::memory_order_relaxed);
        total += n;
        if (n == 0 || off >= sizeof(reasons)) {
            continue;
        }
        const int w = std::snprintf(reasons + off, sizeof(reasons) - off, "%s%s:%llu", off ? "," : "",
            occamy_fa_gemm_reason_names[i], (unsigned long long) n);
        if (w > 0) {
            off += size_t(w);
        }
    }
    std::fprintf(stderr, "occamy_fa_gemm: engaged=%llu fallback=%llu reasons=%s when=%s\n",
        (unsigned long long) occamy_fa_gemm_engaged.load(std::memory_order_relaxed), (unsigned long long) total,
        off ? reasons : "-", when);
    if (occamy_fa_gemm_fused_enabled()) {
        std::fprintf(stderr, "occamy_fa_gemm_fused: calls=%llu softmax_launches=%llu when=%s\n",
            (unsigned long long) occamy_fa_gemm_fused_calls.load(std::memory_order_relaxed),
            (unsigned long long) occamy_fa_gemm_fused_launches.load(std::memory_order_relaxed), when);
    }
    std::fflush(stderr);
}

static void occamy_fa_gemm_atexit() {
    occamy_fa_gemm_print("exit");
}

static void occamy_fa_gemm_first_use() {
    bool expected = false;
    if (!occamy_fa_gemm_used.load(std::memory_order_relaxed) &&
        occamy_fa_gemm_used.compare_exchange_strong(expected, true)) {
        std::atexit(occamy_fa_gemm_atexit);
        occamy_fa_gemm_print("first_use");
    }
}

static void occamy_fa_gemm_fallback(occamy_fa_gemm_reason r) {
    occamy_fa_gemm_fallbacks[r].fetch_add(1, std::memory_order_relaxed);
}

// Called from ggml_backend_cuda_free (defined in fattn.cu, declared in fattn.cuh).
static void occamy_fa_gemm_report_free_impl() {
    if (occamy_fa_gemm_used.load(std::memory_order_relaxed)) {
        occamy_fa_gemm_print("backend_free");
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr int maxwell_gemm_dimension = 256;
static constexpr int maxwell_gemm_query_heads = 16;
static constexpr int maxwell_gemm_kv_heads = 2;
static constexpr int maxwell_gemm_grouped_heads = 8;
static constexpr int maxwell_gemm_tile_queries = 64;
static constexpr int maxwell_gemm_split_keys = 4096;

static __global__ void maxwell_gemm_convert_kv(
        const char * keys, const char * values, float * keys_float, float * values_float,
        int key_count, size_t key_row_stride, size_t value_row_stride) {
    const size_t element_count = size_t(key_count) * maxwell_gemm_dimension;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < element_count;
         index += size_t(blockDim.x) * gridDim.x) {
        const int dimension = index % maxwell_gemm_dimension;
        const size_t key = index / maxwell_gemm_dimension;
        keys_float[index] = __half2float(reinterpret_cast<const half *>(keys + key * key_row_stride)[dimension]);
        values_float[index] = __half2float(reinterpret_cast<const half *>(values + key * value_row_stride)[dimension]);
    }
}

static __global__ void maxwell_gemm_pack_queries(
        const char * input, float * packed, int query_count, int padded_queries,
        size_t query_row_stride, size_t query_head_stride) {
    const size_t element_count = size_t(padded_queries) * maxwell_gemm_query_heads * maxwell_gemm_dimension;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < element_count;
         index += size_t(blockDim.x) * gridDim.x) {
        const int dimension = index % maxwell_gemm_dimension;
        const size_t column = index / maxwell_gemm_dimension;
        const int local_column = column % (maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries);
        const int chunk = column / (maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries);
        const int group = chunk % maxwell_gemm_kv_heads;
        const int query_start = (chunk / maxwell_gemm_kv_heads) * maxwell_gemm_tile_queries;
        const int head = group * maxwell_gemm_grouped_heads + local_column / maxwell_gemm_tile_queries;
        const int query = query_start + local_column % maxwell_gemm_tile_queries;
        packed[index] = query < query_count ? reinterpret_cast<const float *>(input + query * query_row_stride + head * query_head_stride)[dimension] : 0.0f;
    }
}

template<bool maximum>
static __device__ float maxwell_gemm_block_reduce(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = maximum ? fmaxf(value, other) : value + other;
    }
    __shared__ float warp_values[8];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 8 ? warp_values[threadIdx.x] : (maximum ? -FLT_MAX : 0.0f);
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            const float other = __shfl_down_sync(0xffffffffu, value, offset);
            value = maximum ? fmaxf(value, other) : value + other;
        }
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

static __global__ void maxwell_gemm_masked_softmax(
        float * scores, const char * mask, size_t mask_row_stride,
        int key_count, int query_count, int query_start) {
    const int local_query = blockIdx.x;
    const int local_head = blockIdx.y;
    const int query = query_start + local_query;
    float * column = scores + (size_t(local_head) * maxwell_gemm_tile_queries + local_query) * key_count;
    if (query >= query_count) {
        for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
            column[key] = 0.0f;
        }
        return;
    }
    const half * mask_row = mask ? reinterpret_cast<const half *>(mask + query * mask_row_stride) : nullptr;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
        const float value = column[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f);
        if (value > local_max) {
            local_sum = local_sum * __expf(local_max - value) + 1.0f;
            local_max = value;
        } else {
            local_sum += __expf(value - local_max);
        }
    }
    const float maximum = maxwell_gemm_block_reduce<true>(local_max);
    local_sum *= __expf(local_max - maximum);
    const float total = maxwell_gemm_block_reduce<false>(local_sum);
    if (total == 0.0f) {
        for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
            column[key] = 0.0f;
        }
        return;
    }
    const float inverse_sum = 1.0f / total;
    for (int key = threadIdx.x; key < key_count; key += blockDim.x) {
        column[key] = __expf(column[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f) - maximum) * inverse_sum;
    }
}

static __global__ void maxwell_gemm_pv_pointers(
        const float * values, const float * scores, float * partials,
        const float ** left, const float ** right, float ** output, int split_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= split_count) {
        return;
    }
    left[index] = values + size_t(index) * maxwell_gemm_split_keys * maxwell_gemm_dimension;
    right[index] = scores + index * maxwell_gemm_split_keys;
    output[index] = partials + size_t(index) * maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries * maxwell_gemm_dimension;
}

static __global__ void maxwell_gemm_reduce_products(
        const float * partials, float * output, int query_count, int query_start, int head_start, int split_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int element_count = maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries * maxwell_gemm_dimension;
    if (index >= element_count) {
        return;
    }
    const int dimension = index % maxwell_gemm_dimension;
    const int local_query = (index / maxwell_gemm_dimension) % maxwell_gemm_tile_queries;
    const int local_head = index / (maxwell_gemm_dimension * maxwell_gemm_tile_queries);
    if (query_start + local_query >= query_count) {
        return;
    }
    float total = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        total += partials[size_t(split) * element_count + index];
    }
    output[((size_t(query_start) + local_query) * maxwell_gemm_query_heads + head_start + local_head) * maxwell_gemm_dimension + dimension] = total;
}

// WP2: fused mask + softmax, split-local. One block per (head, split, query): the block reads its
// 4096-key slice of the S column once (16 values per thread in registers), adds the F16 mask, and
// writes p = exp(s - m_split) in place (S read once, written once, mask read once). The split max
// m and sum l go to `stats`; normalisation and the cross-split rescale exp(m_split - M) / L are
// applied to the PV partials in maxwell_gemm_reduce_products_fused (flash-decoding LSE combine).
static constexpr int maxwell_gemm_fused_threads = 256;
static constexpr int maxwell_gemm_fused_per_thread = maxwell_gemm_split_keys / maxwell_gemm_fused_threads;

static __global__ void __launch_bounds__(maxwell_gemm_fused_threads) maxwell_gemm_fused_split_softmax(
        float * scores, float2 * stats, const char * mask, size_t mask_row_stride,
        int key_count, int query_count, int query_start, int split_capacity) {
    const int local_head = blockIdx.x;
    const int split = blockIdx.y;
    const int local_query = blockIdx.z;
    const int column_index = local_head * maxwell_gemm_tile_queries + local_query;
    const int query = query_start + local_query;
    const int key_start = split * maxwell_gemm_split_keys;
    const int key_end = min(key_count, key_start + maxwell_gemm_split_keys);
    float * column = scores + size_t(column_index) * key_count;
    if (query >= query_count) {
        for (int key = key_start + threadIdx.x; key < key_end; key += maxwell_gemm_fused_threads) {
            column[key] = 0.0f;
        }
        if (threadIdx.x == 0) {
            stats[size_t(column_index) * split_capacity + split] = make_float2(-FLT_MAX, 0.0f);
        }
        return;
    }
    const half * mask_row = mask ? reinterpret_cast<const half *>(mask + size_t(query) * mask_row_stride) : nullptr;
    float values[maxwell_gemm_fused_per_thread];
    float local_max = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < maxwell_gemm_fused_per_thread; ++i) {
        const int key = key_start + threadIdx.x + i * maxwell_gemm_fused_threads;
        float value = -INFINITY;
        if (key < key_end) {
            value = column[key] + (mask_row ? __half2float(mask_row[key]) : 0.0f);
        }
        values[i] = value;
        local_max = fmaxf(local_max, value);
    }
    __shared__ float warp_max[maxwell_gemm_fused_threads / 32];
    __shared__ float warp_sum[maxwell_gemm_fused_threads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
    }
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();
    float maximum = warp_max[0];
#pragma unroll
    for (int w = 1; w < maxwell_gemm_fused_threads / 32; ++w) {
        maximum = fmaxf(maximum, warp_max[w]);
    }
    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < maxwell_gemm_fused_per_thread; ++i) {
        const int key = key_start + threadIdx.x + i * maxwell_gemm_fused_threads;
        const float p = __expf(values[i] - maximum);
        local_sum += p;
        if (key < key_end) {
            column[key] = p;
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
    }
    if (lane == 0) {
        warp_sum[warp] = local_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < maxwell_gemm_fused_threads / 32; ++w) {
            total += warp_sum[w];
        }
        stats[size_t(column_index) * split_capacity + split] = make_float2(maximum, total);
    }
}

static __global__ void maxwell_gemm_reduce_products_fused(
        const float * partials, const float2 * stats, float * output,
        int query_count, int query_start, int head_start, int split_count, int split_capacity) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int element_count = maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries * maxwell_gemm_dimension;
    if (index >= element_count) {
        return;
    }
    const int dimension = index % maxwell_gemm_dimension;
    const int local_query = (index / maxwell_gemm_dimension) % maxwell_gemm_tile_queries;
    const int local_head = index / (maxwell_gemm_dimension * maxwell_gemm_tile_queries);
    if (query_start + local_query >= query_count) {
        return;
    }
    const float2 * column_stats = stats + size_t(local_head * maxwell_gemm_tile_queries + local_query) * split_capacity;
    float maximum = -FLT_MAX;
    for (int split = 0; split < split_count; ++split) {
        const float2 st = column_stats[split];
        if (st.y > 0.0f) {
            maximum = fmaxf(maximum, st.x);
        }
    }
    float sum = 0.0f;
    float total = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        const float2 st = column_stats[split];
        if (st.y > 0.0f) {
            const float weight = expf(st.x - maximum);
            sum += weight * st.y;
            total += weight * partials[size_t(split) * element_count + index];
        }
    }
    output[((size_t(query_start) + local_query) * maxwell_gemm_query_heads + head_start + local_head) * maxwell_gemm_dimension + dimension] =
        sum > 0.0f ? total / sum : 0.0f;
}

static bool ggml_cuda_flash_attn_ext_maxwell_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_FA_MAXWELL_GEMM");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) {
        return false;
    }
    occamy_fa_gemm_first_use();
#define OFG_FALLBACK(r) do { occamy_fa_gemm_fallback(OFG_##r); return false; } while (0)
    if (ggml_cuda_info().devices[ctx.device].cc != 520) OFG_FALLBACK(cc);
    const ggml_tensor * queries = dst->src[0];
    const ggml_tensor * keys = dst->src[1];
    const ggml_tensor * values = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!queries || !keys || !values) OFG_FALLBACK(nullsrc);
    if (queries->type != GGML_TYPE_F32) OFG_FALLBACK(qtype);
    if (keys->type != GGML_TYPE_F16) OFG_FALLBACK(ktype);
    if (values->type != GGML_TYPE_F16) OFG_FALLBACK(vtype);
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) OFG_FALLBACK(dst);
    if (dst->src[4]) OFG_FALLBACK(sinks);
    if (queries->ne[0] != 256 || keys->ne[0] != 256 || values->ne[0] != 256) OFG_FALLBACK(headdim);
    if (queries->ne[2] != 16 || keys->ne[2] != 2 || values->ne[2] != 2) OFG_FALLBACK(heads);
    if (queries->ne[3] != 1 || keys->ne[3] != 1 || values->ne[3] != 1) OFG_FALLBACK(ne3);
    if (queries->ne[1] < 64) OFG_FALLBACK(q_lt_64);
    if (queries->ne[1] > 1024) OFG_FALLBACK(q_gt_1024);
    if (keys->ne[1] < 4096) OFG_FALLBACK(keys_lt_4096);
    if (keys->ne[1] > 98304) OFG_FALLBACK(keys_gt_98304);
    if (values->ne[1] != keys->ne[1] || dst->ne[0] != 256 || dst->ne[1] != 16 ||
        dst->ne[2] != queries->ne[1] || dst->ne[3] != 1) OFG_FALLBACK(shape);
    if (queries->nb[0] != sizeof(float) || keys->nb[0] != sizeof(half) || values->nb[0] != sizeof(half) ||
        queries->nb[1] % sizeof(float) || queries->nb[2] % sizeof(float) ||
        keys->nb[1] % sizeof(half) || keys->nb[2] % sizeof(half) ||
        values->nb[1] % sizeof(half) || values->nb[2] % sizeof(half)) OFG_FALLBACK(stride);
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < keys->ne[1] || mask->ne[1] < queries->ne[1] ||
                 mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half))) {
        OFG_FALLBACK(mask);
    }
    float scale;
    float max_bias;
    float softcap;
    std::memcpy(&scale, reinterpret_cast<const float *>(dst->op_params) + 0, sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));
    std::memcpy(&softcap, reinterpret_cast<const float *>(dst->op_params) + 2, sizeof(float));
    if (!std::isfinite(scale) || max_bias != 0.0f || softcap != 0.0f || ggml_get_op_params_i32(dst, 4) != 0) {
        OFG_FALLBACK(opparams);
    }
#undef OFG_FALLBACK
    occamy_fa_gemm_engaged.fetch_add(1, std::memory_order_relaxed);
    const int query_count = queries->ne[1];
    const int key_count = keys->ne[1];
    const int padded_queries = ((query_count + maxwell_gemm_tile_queries - 1) / maxwell_gemm_tile_queries) * maxwell_gemm_tile_queries;
    const int key_capacity = 98304;
    const int split_count = (key_count + maxwell_gemm_split_keys - 1) / maxwell_gemm_split_keys;
    const int full_splits = key_count / maxwell_gemm_split_keys;
    const int tail_keys = key_count % maxwell_gemm_split_keys;
    const int split_capacity = key_capacity / maxwell_gemm_split_keys;
    const int packed_columns = maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries;
    const int product_elements = packed_columns * maxwell_gemm_dimension;
    const size_t key_elements = size_t(key_capacity) * maxwell_gemm_dimension;
    const size_t query_elements = size_t(1024) * maxwell_gemm_query_heads * maxwell_gemm_dimension;
    const size_t score_elements = size_t(key_capacity) * packed_columns;
    const size_t partial_elements = size_t(split_capacity) * product_elements;
    const size_t pointer_count = split_capacity;
    const size_t data_bytes = (key_elements * 2 + query_elements + score_elements + partial_elements) * sizeof(float);
    const bool fused = occamy_fa_gemm_fused_enabled();
    const size_t stats_offset = data_bytes + pointer_count * sizeof(float *) * 3;
    const size_t stats_bytes = fused ? size_t(packed_columns) * split_capacity * sizeof(float2) : 0;
    const size_t scratch_bytes = stats_offset + stats_bytes;
    ggml_cuda_pool_alloc<char> scratch(ctx.pool(), scratch_bytes);
    float * keys_float = reinterpret_cast<float *>(scratch.get());
    float * values_float = keys_float + key_elements;
    float * packed_queries = values_float + key_elements;
    float * scores = packed_queries + query_elements;
    float * partials = scores + score_elements;
    const float ** pv_left = reinterpret_cast<const float **>(scratch.get() + data_bytes);
    const float ** pv_right = pv_left + pointer_count;
    float ** pv_output = reinterpret_cast<float **>(scratch.get() + data_bytes + 2 * pointer_count * sizeof(float *));
    float2 * softmax_stats = fused ? reinterpret_cast<float2 *>(scratch.get() + stats_offset) : nullptr;
    if (fused) {
        occamy_fa_gemm_fused_calls.fetch_add(1, std::memory_order_relaxed);
    }
    const cudaStream_t stream = ctx.stream();
    const cublasHandle_t handle = ctx.cublas_handle();
    static std::atomic<uint64_t> calls{0};
    const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (call == 1) {
        GGML_LOG_INFO("maxwell_gemm: version=4 softmax=%s kv_staging_heads=1 call=%llu q=%d k=%d q_heads=16 kv_heads=2 d=256 mask_stride=%zu key_capacity=%d splits=%d tail=%d scratch_bytes=%zu pool_bytes=%zu cublas_workspace_bytes=%zu\n",
            fused ? "fused_split_lse" : "online", (unsigned long long) call, query_count, key_count, mask ? mask->nb[1] : 0, key_capacity, split_count, tail_keys,
            scratch_bytes, scratch.actual_size, ctx.cublas_workspace_sizes[ctx.device]);
    }
    maxwell_gemm_pack_queries<<<4096, 256, 0, stream>>>(
        static_cast<const char *>(queries->data), packed_queries, query_count, padded_queries, queries->nb[1], queries->nb[2]);
    maxwell_gemm_pv_pointers<<<1, 256, 0, stream>>>(
        values_float, scores, partials, pv_left, pv_right, pv_output, split_count);
    CUDA_CHECK(cudaGetLastError());
    const float one = 1.0f;
    const float zero = 0.0f;
    for (int group = 0; group < maxwell_gemm_kv_heads; ++group) {
        maxwell_gemm_convert_kv<<<65535, 256, 0, stream>>>(
            static_cast<const char *>(keys->data) + group * keys->nb[2],
            static_cast<const char *>(values->data) + group * values->nb[2],
            keys_float, values_float, key_count, keys->nb[1], values->nb[1]);
        CUDA_CHECK(cudaGetLastError());
        for (int query_start = 0; query_start < padded_queries; query_start += maxwell_gemm_tile_queries) {
            const int chunk = (query_start / maxwell_gemm_tile_queries) * maxwell_gemm_kv_heads + group;
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                key_count, packed_columns, maxwell_gemm_dimension, &scale,
                keys_float, maxwell_gemm_dimension,
                packed_queries + size_t(chunk) * product_elements, maxwell_gemm_dimension,
                &zero, scores, key_count));
            if (fused) {
                maxwell_gemm_fused_split_softmax<<<dim3(maxwell_gemm_grouped_heads, split_count, maxwell_gemm_tile_queries),
                    maxwell_gemm_fused_threads, 0, stream>>>(
                    scores, softmax_stats, mask ? static_cast<const char *>(mask->data) : nullptr, mask ? mask->nb[1] : 0,
                    key_count, query_count, query_start, split_capacity);
                occamy_fa_gemm_fused_launches.fetch_add(1, std::memory_order_relaxed);
            } else {
                maxwell_gemm_masked_softmax<<<dim3(maxwell_gemm_tile_queries, maxwell_gemm_grouped_heads), 256, 0, stream>>>(
                    scores, mask ? static_cast<const char *>(mask->data) : nullptr, mask ? mask->nb[1] : 0,
                    key_count, query_count, query_start);
            }
            CUDA_CHECK(cudaGetLastError());
            if (full_splits > 0) {
                CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    maxwell_gemm_dimension, packed_columns, maxwell_gemm_split_keys, &one,
                    pv_left, maxwell_gemm_dimension,
                    pv_right, key_count, &zero, pv_output, maxwell_gemm_dimension, full_splits));
            }
            if (tail_keys > 0) {
                const int key_start = full_splits * maxwell_gemm_split_keys;
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    maxwell_gemm_dimension, packed_columns, tail_keys, &one,
                    values_float + size_t(key_start) * maxwell_gemm_dimension, maxwell_gemm_dimension,
                    scores + key_start, key_count, &zero,
                    partials + size_t(full_splits) * product_elements, maxwell_gemm_dimension));
            }
            if (fused) {
                maxwell_gemm_reduce_products_fused<<<(product_elements + 255) / 256, 256, 0, stream>>>(
                    partials, softmax_stats, static_cast<float *>(dst->data), query_count, query_start,
                    group * maxwell_gemm_grouped_heads, split_count, split_capacity);
            } else {
                maxwell_gemm_reduce_products<<<(product_elements + 255) / 256, 256, 0, stream>>>(
                    partials, static_cast<float *>(dst->data), query_count, query_start, group * maxwell_gemm_grouped_heads, split_count);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}

#else

static bool ggml_cuda_flash_attn_ext_maxwell_gemm(ggml_backend_cuda_context &, ggml_tensor *) {
    return false;
}

#endif
