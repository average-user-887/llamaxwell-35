#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

constexpr int gdn_cols_per_warp = 4;

// sm52 experiment 12 Oct 2026 (GDN retile): columns per warp is selectable at runtime, separately for decode
// (n_tokens == 1) and prefill. Valid values 1, 2, 4; default 4 (upstream layout).
//   GGML_CUDA_GDN_COLS=<n>            both
//   GGML_CUDA_GDN_COLS=<dec>,<pre>    decode,prefill
//   GGML_CUDA_GDN_COLS_DECODE=<n>, GGML_CUDA_GDN_COLS_PREFILL=<n>   override either one
static int gdn_cols_sanitize(int v, int fallback) {
    return (v == 1 || v == 2 || v == 4) ? v : fallback;
}

struct gdn_cols_cfg {
    int decode;
    int prefill;
};

static gdn_cols_cfg gdn_cols_read_env() {
    gdn_cols_cfg c = { gdn_cols_per_warp, gdn_cols_per_warp };
    if (const char * e = getenv("GGML_CUDA_GDN_COLS")) {
        c.decode = c.prefill = gdn_cols_sanitize(atoi(e), gdn_cols_per_warp);
        if (const char * comma = strchr(e, ',')) {
            c.prefill = gdn_cols_sanitize(atoi(comma + 1), gdn_cols_per_warp);
        }
    }
    if (const char * e = getenv("GGML_CUDA_GDN_COLS_DECODE")) {
        c.decode = gdn_cols_sanitize(atoi(e), c.decode);
    }
    if (const char * e = getenv("GGML_CUDA_GDN_COLS_PREFILL")) {
        c.prefill = gdn_cols_sanitize(atoi(e), c.prefill);
    }
    return c;
}

static int gdn_cols_from_env(const bool decode) {
    static const gdn_cols_cfg cfg = gdn_cols_read_env();
    return decode ? cfg.decode : cfg.prefill;
}

static bool gdn_smem_prefill_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_GDN_SMEM_PREFILL_TILE8");
        return e && strcmp(e, "1") == 0;
    }();
    return enabled;
}

// Cooperatively stage tiles of (q, k, v, exp(g), beta) in shared memory for long prefill.
// Preserves the exact per-lane FP32 reduction order of gated_delta_net_cuda.
template <int S_v, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp, int TILE_T = 8>
__global__ void __launch_bounds__(32 * 4, 6)
gated_delta_net_smem_prefill_cuda(const float * __restrict__ q,
                                  const float * __restrict__ k,
                                  const float * __restrict__ v,
                                  const float * __restrict__ g,
                                  const float * __restrict__ beta,
                                  const float *              curr_state,
                                  float *                    dst,
                                  float *                    state,
                                  int64_t                    H,
                                  int64_t                    n_tokens,
                                  int64_t                    n_seqs,
                                  int64_t                    sq1,
                                  int64_t                    sq2,
                                  int64_t                    sq3,
                                  int64_t                    sv1,
                                  int64_t                    sv2,
                                  int64_t                    sv3,
                                  int64_t                    sb1,
                                  int64_t                    sb2,
                                  int64_t                    sb3,
                                  const uint3                neqk1_magic,
                                  const uint3                rq3_magic,
                                  float                      scale,
                                  int64_t                    state_slot_stride,
                                  int                        K) {
    static_assert(S_v == 128, "smem prefill requires S_v == 128");
    constexpr int warp_size     = 32;
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(lanes_per_col >= 1 && (warp_size % cols_per_warp) == 0, "cols_per_warp must divide warp");
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    const int lane         = threadIdx.x;
    const int warp_id      = threadIdx.y;
    const int num_warps    = blockDim.y;
    const int col_in_warp  = lane / lanes_per_col;
    const int lane_in_col  = lane - col_in_warp * lanes_per_col;
    const int cols_per_cta = num_warps * cols_per_warp;
    const int col_cta_base = blockIdx.z * cols_per_cta;
    const int local_col    = warp_id * cols_per_warp + col_in_warp;
    const int col          = col_cta_base + local_col;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_base    = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base    = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base    = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base   = sequence * sb3 + h_idx * sb1;
    const float * g_base    = g + gb_base;
    const float * beta_base = beta + gb_base;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    alignas(16) __shared__ float s_q[TILE_T][S_v];
    alignas(16) __shared__ float s_k[TILE_T][S_v];
    __shared__ float s_v[TILE_T][16];
    __shared__ float s_g[TILE_T];
    __shared__ float s_beta[TILE_T];

    float s_shard[rows_per_lane];
    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    // Swizzle indices for conflict-free STS.32 and LDS.128 when cols_per_warp == 4.
    const int ld_r     = lane >> 1;
    const int ld_hi    = lane & 1;
    const int st_l0    = ld_hi ? 5 : 0;
    const int st_l1    = ld_hi ? 4 : 1;
    const int st_l2    = ld_hi ? 7 : 2;
    const int st_l3    = ld_hi ? 6 : 3;
    const int st_r01   = ld_r ^ (ld_hi ? 8 : 0);
    const int st_r23   = ld_r ^ (ld_hi ? 12 : 4);
    const int rd_l     = lane_in_col ^ (lane_in_col >> 2);
    const int rd_c_xor = rd_l >> 1;

    for (int64_t t_base = 0; t_base < n_tokens; t_base += TILE_T) {
        const int tile_len = (t_base + TILE_T <= n_tokens) ? TILE_T : (int) (n_tokens - t_base);

        for (int tt = warp_id; tt < tile_len; tt += num_warps) {
            const int64_t t  = t_base + tt;
            const float4  qv = reinterpret_cast<const float4 *>(q_base + t * sq2)[lane];
            const float4  kv = reinterpret_cast<const float4 *>(k_base + t * sq2)[lane];
            if constexpr (cols_per_warp == 4) {
                float * sq_tt = s_q[tt];
                float * sk_tt = s_k[tt];
                sq_tt[st_l0 * 16 + st_r01] = qv.x;
                sq_tt[st_l1 * 16 + st_r01] = qv.y;
                sq_tt[st_l2 * 16 + st_r23] = qv.z;
                sq_tt[st_l3 * 16 + st_r23] = qv.w;
                sk_tt[st_l0 * 16 + st_r01] = kv.x;
                sk_tt[st_l1 * 16 + st_r01] = kv.y;
                sk_tt[st_l2 * 16 + st_r23] = kv.z;
                sk_tt[st_l3 * 16 + st_r23] = kv.w;
            } else {
                reinterpret_cast<float4 *>(s_q[tt])[lane] = qv;
                reinterpret_cast<float4 *>(s_k[tt])[lane] = kv;
            }
            if (lane < cols_per_cta) {
                s_v[tt][lane] = v_base[t * sv2 + col_cta_base + lane];
            }
            if (lane == 0) {
                s_g[tt]    = expf(g_base[t * sb2]);
                s_beta[tt] = beta_base[t * sb2];
            }
        }
        __syncthreads();

        for (int tt = 0; tt < tile_len; ++tt) {
            const float g_val    = s_g[tt];
            const float beta_val = s_beta[tt];

            float k_reg[rows_per_lane];
            float kv_shard = 0.0f;
            if constexpr (cols_per_warp == 4) {
                const float4 * sk_row = reinterpret_cast<const float4 *>(s_k[tt] + rd_l * 16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 k4 = sk_row[c ^ rd_c_xor];
                    k_reg[c * 4 + 0] = k4.x;
                    k_reg[c * 4 + 1] = k4.y;
                    k_reg[c * 4 + 2] = k4.z;
                    k_reg[c * 4 + 3] = k4.w;
                    kv_shard += s_shard[c * 4 + 0] * k4.x;
                    kv_shard += s_shard[c * 4 + 1] * k4.y;
                    kv_shard += s_shard[c * 4 + 2] * k4.z;
                    kv_shard += s_shard[c * 4 + 3] * k4.w;
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * lanes_per_col + lane_in_col;
                    k_reg[r] = s_k[tt][i];
                    kv_shard += s_shard[r] * k_reg[r];
                }
            }

            const float kv_col    = warp_reduce_sum<lanes_per_col>(kv_shard);
            const float delta_col = (s_v[tt][local_col] - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
            if constexpr (cols_per_warp == 4) {
                const float4 * sq_row = reinterpret_cast<const float4 *>(s_q[tt] + rd_l * 16);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float4 q4 = sq_row[c ^ rd_c_xor];
                    s_shard[c * 4 + 0] = g_val * s_shard[c * 4 + 0] + k_reg[c * 4 + 0] * delta_col;
                    attn_partial += s_shard[c * 4 + 0] * q4.x;
                    s_shard[c * 4 + 1] = g_val * s_shard[c * 4 + 1] + k_reg[c * 4 + 1] * delta_col;
                    attn_partial += s_shard[c * 4 + 1] * q4.y;
                    s_shard[c * 4 + 2] = g_val * s_shard[c * 4 + 2] + k_reg[c * 4 + 2] * delta_col;
                    attn_partial += s_shard[c * 4 + 2] * q4.z;
                    s_shard[c * 4 + 3] = g_val * s_shard[c * 4 + 3] + k_reg[c * 4 + 3] * delta_col;
                    attn_partial += s_shard[c * 4 + 3] * q4.w;
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * lanes_per_col + lane_in_col;
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[r] * s_q[tt][i];
                }
            }

            const float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);
            if (lane_in_col == 0) {
                s_v[tt][local_col] = attn_col * scale;
            }

            if constexpr (keep_rs_t) {
                const int64_t target_slot = n_tokens - 1 - (t_base + tt);
                if (target_slot >= 0 && target_slot < K) {
                    float * dst_state = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * lanes_per_col + lane_in_col;
                        dst_state[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
        __syncthreads();

        // Each warp flushes its own tt slots; next tile's load writes the same tt slots per warp.
        for (int tt = warp_id; tt < tile_len; tt += num_warps) {
            if (lane < cols_per_cta) {
                attn_data[(t_base + tt) * S_v * H + col_cta_base + lane] = s_v[tt][lane];
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;
    static_assert(lanes_per_col >= 1 && (warp_size % cols_per_warp) == 0, "cols_per_warp must divide the warp");
    if (col >= S_v) { // whole lane segment exits together: no divergence inside a column's reduction
        return;
    }

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, int cols_per_warp>
static void launch_gated_delta_net_cols(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel); S_v is a power of two >= 16 and
    // cols_per_warp*num_warps <= 16, so the rounded-up grid below is exact and the kernel's col guard is a no-op
    const int cols_per_cta = cols_per_warp * num_warps;
    GGML_ASSERT(cols_per_cta <= S_v);
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_cta - 1) / cols_per_cta);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            if constexpr (!KDA && cols_per_warp == 4) {
                // Released Occamy prefill geometry. Other shapes keep their original kernel.
                if (n_tokens >= 64 && gdn_smem_prefill_enabled() &&
                        warp_size == 32 && num_warps == 4 && H == 32 && neqk1 == 16 && n_seqs == 1 &&
                        (((uintptr_t) q_d | (uintptr_t) k_d) & 15u) == 0 && ((sq1 | sq2 | sq3) & 3) == 0 &&
                        ggml_cuda_info().devices[ggml_cuda_get_device()].cc == 520) {
                    static const bool reported = [] {
                        GGML_LOG_INFO("gdn_smem_prefill: active tile=8 heads=32 qk_heads=16 cols=4 snapshots=%d\n",
                                      keep_rs_t ? 1 : 0);
                        return true;
                    }();
                    (void) reported;
                    ggml_cuda_kernel_launch(gated_delta_net_smem_prefill_cuda<128, keep_rs_t, cols_per_warp>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, cols_per_warp>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int cols = gdn_cols_from_env(n_tokens == 1);
    switch (cols) {
        case 1:
            launch_gated_delta_net_cols<KDA, keep_rs_t, 1>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            break;
        case 2:
            launch_gated_delta_net_cols<KDA, keep_rs_t, 2>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            break;
        default:
            launch_gated_delta_net_cols<KDA, keep_rs_t, 4>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_direct(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K, const int32_t *rows, size_t source_stride) {
    const uint32_t h_idx    = blockIdx.x;


    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;
    static_assert(lanes_per_col >= 1 && (warp_size % cols_per_warp) == 0, "cols_per_warp must divide the warp");
    if (col >= S_v) { // whole lane segment exits together: no divergence inside a column's reduction
        return;
    }

    const float*source_base=curr_state;float*state_base=state;
    float before[2][rows_per_lane];
#pragma unroll
    for(int seq=0;seq<2;++seq){
        const float*src=source_base+(size_t)rows[seq]*source_stride+h_idx*S_v*S_v+col*S_v;
#pragma unroll
        for(int r=0;r<rows_per_lane;++r)before[seq][r]=src[r*lanes_per_col+lane_in_col];
    }
    __syncthreads(); // both aliases are fully read before either destination is written
    for(uint32_t sequence=0;sequence<2;++sequence){
        float*state=state_base;const float*curr_state=source_base;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = before[sequence][r];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
    } // sequence
}

bool ggml_cuda_gdn_direct_pair_supported(){return gdn_cols_from_env(true)==2;}
void ggml_cuda_gdn_direct_pair(ggml_backend_cuda_context&ctx,ggml_tensor*dst,const float*source,
        const int32_t*ids,size_t source_stride,float*state,float*attn_override){
    const ggml_tensor*q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*g=dst->src[3],*b=dst->src[4];
    GGML_ASSERT(v->ne[0]==128 && v->ne[1]==32 && v->ne[2]==1 && v->ne[3]==2 && g->ne[0]==1 && ggml_get_op_params_i32(dst,0)==1);
    const ggml_cuda_kernel_launch_params launch_params(dim3(32,1,16),dim3(32,4),0,ctx.stream());
    ggml_cuda_kernel_launch(gated_delta_net_direct<128,false,false,2>,launch_params,
        (const float*)q->data,(const float*)k->data,(const float*)v->data,(const float*)g->data,(const float*)b->data,
        source,attn_override?attn_override:(float*)dst->data,state,int64_t(32),int64_t(1),int64_t(2),
        int64_t(q->nb[1]/4),int64_t(q->nb[2]/4),int64_t(q->nb[3]/4),
        int64_t(v->nb[1]/4),int64_t(v->nb[2]/4),int64_t(v->nb[3]/4),
        int64_t(b->nb[1]/4),int64_t(b->nb[2]/4),int64_t(b->nb[3]/4),
        init_fastdiv_values(q->ne[1]),init_fastdiv_values(v->ne[3]/q->ne[3]),1.0f/sqrtf(128.0f),int64_t(2*32*128*128),1,ids,source_stride);
    CUDA_CHECK(cudaGetLastError());
}
