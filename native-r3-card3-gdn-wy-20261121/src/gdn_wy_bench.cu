// Standalone Chunked WY Gated Delta Net (GDN) prefill benchmark and FP64 validator for sm_52.
// Shape target: Occamy 35B (S_v=128, H=32, neqk1=16, KDA=false, n_seqs=1, chunk B=64).

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                              \
            exit(2);                                                                         \
        }                                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                                   \
    do {                                                                                     \
        cublasStatus_t st__ = (call);                                                        \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                 \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__, __FILE__, __LINE__);    \
            exit(3);                                                                         \
        }                                                                                    \
    } while (0)

static constexpr int SV = 128;
static constexpr int NH = 32;
static constexpr int HK = 16;
static constexpr int CHUNK_B = 64;

// --- Minimal helpers matching ggml-cuda fastdiv / warp reduce ---
struct FastDiv {
    uint3 val;
};

static FastDiv make_fastdiv(int64_t d) {
    uint3 res{0, 0, 0};
    if (d <= 1) {
        res.x = 0;
        res.y = 0;
        res.z = 1;
        return FastDiv{res};
    }
    // Simple exact helper for small power-of-two/positive ints matching ggml fastdiv:
    // Use uint64_t magic in host test; below we pass H_k=16 and rq3=1 via simple modulo/div helper
    uint32_t l = 0;
    while ((1ULL << l) < (uint64_t)d) ++l;
    uint64_t m = ((1ULL << 32) * ((1ULL << l) - (uint64_t)d)) / (uint64_t)d + 1;
    res.x = (uint32_t)m;
    res.y = l;
    res.z = (uint32_t)d;
    return FastDiv{res};
}

__device__ __forceinline__ uint32_t fastdiv_u32(uint32_t n, uint3 fd) {
    if (fd.z <= 1) return n;
    uint32_t t = __umulhi(fd.x, n);
    return (t + n) >> fd.y;
}

__device__ __forceinline__ uint32_t fastmod_u32(uint32_t n, uint3 fd) {
    if (fd.z <= 1) return 0;
    return n - fastdiv_u32(n, fd) * fd.z;
}

template <int width>
__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, offset, 32);
    }
    return x;
}

// --- Stock gated_delta_net_cuda (KDA=false, S_v=128, cols_per_warp=4 matching GGML_CUDA_GDN_COLS=2,4) ---
template <bool keep_rs_t, int cols_per_warp = 4>
__global__ void __launch_bounds__(128, 2)
stock_gdn_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float * __restrict__ dst,
        float * __restrict__ state,
        int64_t H,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic,
        const uint3 rq3_magic,
        float scale,
        int64_t state_slot_stride,
        int K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size     = 32;
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = SV / lanes_per_col;

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col         = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;
    if (col >= SV) return;

    const uint32_t iq1 = fastmod_u32(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv_u32(sequence, rq3_magic);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * SV;
    state      += (sequence * H + h_idx) * SV * SV;
    curr_state += sequence * H * SV * SV + h_idx * SV * SV + col * SV;

    float s_shard[rows_per_lane];
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
        const float   beta_val  = beta[gb_offset];
        const float   g_val     = expf(g[gb_offset]);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        float kv_col    = warp_reduce_sum<lanes_per_col>(kv_shard);
        float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            attn_data[col] = attn_col * scale;
        }
        attn_data += SV * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int)n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * slot_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    slot_state[col * SV + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * SV + i] = s_shard[r];
        }
    }
}

// --- Chunked WY Kernels (B = 64, S_v = 128) ---
// Workspace layout for n_chunks chunks (each batch = (chunk, head) in chunk-major order: idx = c * H + h):
//   q_pack     [n_chunks * H, 64, 128]
//   k_pack     [n_chunks * H, 64, 128]
//   v_beta     [n_chunks * H, 64, 128]
//   k_beta_g   [n_chunks * H, 64, 128]
//   k_tilde    [n_chunks * H, 64, 128]
//   wq_hat     [n_chunks * H, 128, 128]   (first 64 rows = W, next 64 rows = Q_hat per idx)
//   ws_o       [n_chunks * H, 128, 128]   (first 64 rows = WS, next 64 rows = O_inter -> final O per idx)
//   delta      [n_chunks * H, 64, 128]
//   gram_mat   [n_chunks * H, 64, 64]     (first K_gram -> A = (I+L)^{-1}, then reused for Q_gram -> M)
//   g_cum      [n_chunks * H, 64]
//   beta_buf   [n_chunks * H, 64]
//   chunk_dec  [n_chunks * H]
//   s_work     [H, 128, 128]

// Kernel 1: Prefix-sum g and pack Q, K, V_beta, K_beta_g, K_tilde, Q_hat.
// Grid: dim3(H, n_chunks), Block: 128 threads (4 warps).
// 1 chunk * 1 head = 64 tokens * 128 dims.
__global__ void __launch_bounds__(128, 4)
wy_prepack_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        float * __restrict__ q_pack,
        float * __restrict__ k_pack,
        float * __restrict__ v_beta,
        float * __restrict__ k_beta_g,
        float * __restrict__ k_tilde,
        float * __restrict__ wq_hat,
        double * __restrict__ g_cum,
        float * __restrict__ beta_buf,
        float * __restrict__ chunk_dec,
        int H, int n_chunks,
        int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2,
        const uint3 neqk1_magic) {
    const int h   = blockIdx.x;
    const int c   = blockIdx.y;
    const int idx = c * H + h;
    const int tid = threadIdx.x; // 0..127

    __shared__ double sg[CHUNK_B];
    __shared__ float sb[CHUNK_B];

    // Load 64 g and beta values and prefix-sum g in double using threads 0..63
    if (tid < CHUNK_B) {
        const int t = c * CHUNK_B + tid;
        const int64_t off = (int64_t)t * sb2 + (int64_t)h * sb1;
        sg[tid] = (double)g[off];
        sb[tid] = beta[off];
    }
    __syncthreads();

    if (tid == 0) {
        double acc = 0.0;
        for (int j = 0; j < CHUNK_B; ++j) {
            acc += sg[j];
            sg[j] = acc;
        }
        chunk_dec[idx] = (float)exp(acc);
    }
    __syncthreads();

    if (tid < CHUNK_B) {
        g_cum[(size_t)idx * CHUNK_B + tid]    = sg[tid];
        beta_buf[(size_t)idx * CHUNK_B + tid] = sb[tid];
    }

    const double G_end = sg[CHUNK_B - 1];
    const uint32_t iq1 = fastmod_u32((uint32_t)h, neqk1_magic);

    const size_t tile64  = (size_t)idx * CHUNK_B * SV;
    const size_t tile128 = (size_t)idx * 128 * SV; // W in [0..63], Q_hat in [64..127]

    float * qp = q_pack + tile64;
    float * kp = k_pack + tile64;
    float * vb = v_beta + tile64;
    float * kb = k_beta_g + tile64;
    float * kt = k_tilde + tile64;
    float * qh = wq_hat + tile128 + CHUNK_B * SV;

    for (int j = 0; j < CHUNK_B; ++j) {
        const int t = c * CHUNK_B + j;
        const float q_val = q[(int64_t)t * sq2 + (int64_t)iq1 * sq1 + tid];
        const float k_val = k[(int64_t)t * sq2 + (int64_t)iq1 * sq1 + tid];
        const float v_val = v[(int64_t)t * sv2 + (int64_t)h * sv1 + tid];

        const double Gj      = sg[j];
        const float b_val    = sb[j];
        const float decay_j  = (float)exp(Gj);
        const float end_dec  = (float)exp(fmin(G_end - Gj, 80.0));

        const size_t row_off = (size_t)j * SV + tid;
        qp[row_off] = q_val;
        kp[row_off] = k_val;
        vb[row_off] = b_val * v_val;
        kb[row_off] = b_val * decay_j * k_val;
        kt[row_off] = end_dec * k_val;
        qh[row_off] = decay_j * q_val;
    }
}

// Kernel 2: Form L_{j,r} = beta_j * exp(G_j - G_r) * K_gram[j,r] for j > r, and invert (I + L)^{-1} in place.
// Grid: H * n_chunks, Block: 64 threads.
__global__ void __launch_bounds__(64, 4)
wy_invert_l_kernel(
        float * __restrict__ gram_A,
        const double * __restrict__ g_cum,
        const float * __restrict__ beta_buf) {
    const int idx = blockIdx.x;
    const int r   = threadIdx.x; // 0..63 column

    __shared__ float sM[CHUNK_B][CHUNK_B + 1];
    __shared__ float sRow[2][CHUNK_B + 1];

    float * tile = gram_A + (size_t)idx * CHUNK_B * CHUNK_B;
    const double Gr = g_cum[(size_t)idx * CHUNK_B + r];

    for (int j = 0; j < CHUNK_B; ++j) {
        float val = 0.0f;
        if (j == r) {
            val = 1.0f;
        } else if (j > r) {
            const double Gj = g_cum[(size_t)idx * CHUNK_B + j];
            const float bj  = beta_buf[(size_t)idx * CHUNK_B + j];
            val = bj * (float)exp(fmin(Gj - Gr, 80.0)) * tile[(size_t)j * CHUNK_B + r];
        }
        sM[j][r] = val;
    }
    __syncthreads();

    sRow[1][r] = sM[1][r];
    __syncthreads();

    for (int j = 1; j < CHUNK_B; ++j) {
        const int cur = j & 1;
        const float next_L = (j + 1 < CHUNK_B) ? sM[j + 1][r] : 0.0f;
        if (r < j) {
            float val = -sRow[cur][r];
            for (int m = r + 1; m < j; ++m) {
                val = fmaf(-sRow[cur][m], sM[m][r], val);
            }
            sM[j][r] = val;
        }
        if (j + 1 < CHUNK_B) {
            sRow[cur ^ 1][r] = next_L;
        }
        __syncthreads();
    }

    for (int j = 0; j < CHUNK_B; ++j) {
        tile[(size_t)j * CHUNK_B + r] = sM[j][r];
    }
}

// Kernel 3: Apply causal decay mask to Q_gram -> M:
//   M[j,r] = exp(G_j - G_r) * Q_gram[j,r] for j >= r, 0 for j < r.
__global__ void __launch_bounds__(256, 4)
wy_mask_qgram_kernel(
        float * __restrict__ gram_M,
        const double * __restrict__ g_cum) {
    const int idx = blockIdx.x;
    float * tile = gram_M + (size_t)idx * CHUNK_B * CHUNK_B;
    const double * gc = g_cum + (size_t)idx * CHUNK_B;

    __shared__ double sg[CHUNK_B];
    if (threadIdx.x < CHUNK_B) sg[threadIdx.x] = gc[threadIdx.x];
    __syncthreads();

#pragma unroll
    for (int p = threadIdx.x; p < CHUNK_B * CHUNK_B; p += 256) {
        const int j = p >> 6;
        const int r = p & 63;
        float v = 0.0f;
        if (j >= r) {
            v = (float)exp(fmin(sg[j] - sg[r], 80.0)) * tile[p];
        }
        tile[p] = v;
    }
}

// Kernel 4 (inside chunk c loop, one launch per chunk across all H=32 heads):
//   - Delta_c = U_c - WS_c (tile 32 x 64 x 128)
//   - s_work[h] *= chunk_dec[c*H + h] (32 x 128 x 128)
// Grid: H (=32) * 32? Wait: s_work is 16,384 floats/head, v_beta/WS is 8,192 floats/head.
// Grid: dim3(H, 64), Block: 256 threads -> each CTA handles 256 floats of s_work (1 loop) and 128 floats of Delta!
__global__ void __launch_bounds__(256, 4)
wy_chunk_epilogue_kernel(
        const float * __restrict__ u_c,
        const float * __restrict__ ws_o_c,
        const float * __restrict__ chunk_dec_c,
        float * __restrict__ delta_c,
        float * __restrict__ s_work,
        int H) {
    const int h   = blockIdx.x;
    const int blk = blockIdx.y; // 0..63
    const int tid = threadIdx.x;

    const float dec = chunk_dec_c[h];
    // Update s_work: 16384 floats per head = 64 * 256
    const size_t s_off = (size_t)h * SV * SV + (size_t)blk * 256 + tid;
    s_work[s_off] *= dec;

    // Update Delta for blk < 32 (8192 = 32 * 256 floats)
    if (blk < 32) {
        const size_t d_idx  = (size_t)blk * 256 + tid;
        const size_t u_off  = (size_t)h * CHUNK_B * SV + d_idx;
        const size_t ws_off = (size_t)h * 128 * SV + d_idx; // WS occupies first 64 rows
        delta_c[u_off] = u_c[u_off] - ws_o_c[ws_off];
    }
}

// Kernel 5: Unpack O_inter from ws_o ([n_chunks * H, 128, 128], O at rows 64..127) to dst [SV, H, n_tokens].
// Grid: dim3(H, n_chunks), Block: 128 threads.
__global__ void __launch_bounds__(128, 4)
wy_unpack_out_kernel(
        const float * __restrict__ ws_o,
        float * __restrict__ dst,
        int H,
        float scale) {
    const int h   = blockIdx.x;
    const int c   = blockIdx.y;
    const int idx = c * H + h;
    const int tid = threadIdx.x; // 0..127

    const float * src = ws_o + (size_t)idx * 128 * SV + CHUNK_B * SV + tid;
    for (int j = 0; j < CHUNK_B; ++j) {
        const int t = c * CHUNK_B + j;
        dst[((size_t)t * H + h) * SV + tid] = src[(size_t)j * SV] * scale;
    }
}

struct WyWorkspace {
    float * d_ptr = nullptr;
    size_t  bytes = 0;
    float * q_pack, * k_pack, * v_beta, * k_beta_g, * k_tilde;
    float * wq_hat, * ws_o, * delta, * gram_mat, * beta_buf, * chunk_dec, * s_work;
    double * g_cum;

    void init(int H, int max_chunks) {
        size_t nb = (size_t)max_chunks * H;
        size_t n_floats = 0;
        auto alloc_off = [&](size_t cnt) {
            size_t off = (n_floats + 31) & ~size_t(31);
            n_floats = off + cnt;
            return off;
        };
        size_t o_q    = alloc_off(nb * CHUNK_B * SV);
        size_t o_k    = alloc_off(nb * CHUNK_B * SV);
        size_t o_vb   = alloc_off(nb * CHUNK_B * SV);
        size_t o_kbg  = alloc_off(nb * CHUNK_B * SV);
        size_t o_kt   = alloc_off(nb * CHUNK_B * SV);
        size_t o_wqh  = alloc_off(nb * 128 * SV);
        size_t o_wso  = alloc_off(nb * 128 * SV);
        size_t o_del  = alloc_off(nb * CHUNK_B * SV);
        size_t o_gram = alloc_off(nb * CHUNK_B * CHUNK_B);
        size_t o_gcum = alloc_off(nb * CHUNK_B * 2); // double
        size_t o_beta = alloc_off(nb * CHUNK_B);
        size_t o_cdec = alloc_off(nb);
        size_t o_sw   = alloc_off((size_t)H * SV * SV);

        bytes = n_floats * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
        q_pack   = d_ptr + o_q;
        k_pack   = d_ptr + o_k;
        v_beta   = d_ptr + o_vb;
        k_beta_g = d_ptr + o_kbg;
        k_tilde  = d_ptr + o_kt;
        wq_hat   = d_ptr + o_wqh;
        ws_o     = d_ptr + o_wso;
        delta    = d_ptr + o_del;
        gram_mat = d_ptr + o_gram;
        g_cum    = reinterpret_cast<double *>(d_ptr + o_gcum);
        beta_buf = d_ptr + o_beta;
        chunk_dec= d_ptr + o_cdec;
        s_work   = d_ptr + o_sw;
    }
    void destroy() {
        if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
        d_ptr = nullptr;
    }
};

static void launch_chunked_wy(
        cublasHandle_t cublas,
        WyWorkspace & ws,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_in_d,
        float * dst_d, float * state_out_d,
        int H, int H_k, int n_tokens, int K,
        cudaStream_t stream) {
    const int64_t sq1 = SV, sq2 = SV * H_k, sq3 = SV * H_k * n_tokens;
    const int64_t sv1 = SV, sv2 = SV * H,   sv3 = SV * H * n_tokens;
    const int64_t sb1 = 1,  sb2 = H,        sb3 = H * n_tokens;
    const FastDiv fd_k = make_fastdiv(H_k);
    const FastDiv fd_1 = make_fastdiv(1);
    const float scale = 1.0f / sqrtf((float)SV);

    const int n_chunks = std::max(0, (n_tokens - K) / CHUNK_B);
    const int t_wy     = n_chunks * CHUNK_B;

    if (n_chunks > 0) {
        const int n_batch = H * n_chunks;
        const float one = 1.0f, zero = 0.0f;

        CHECK_CUDA(cudaMemcpyAsync(ws.s_work, s_in_d, (size_t)H * SV * SV * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // 1. Prepack Q, K, V_beta, K_beta_g, K_tilde, Q_hat
        wy_prepack_kernel<<<dim3(H, n_chunks), 128, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d,
            ws.q_pack, ws.k_pack, ws.v_beta, ws.k_beta_g, ws.k_tilde, ws.wq_hat,
            ws.g_cum, ws.beta_buf, ws.chunk_dec,
            H, n_chunks, sq1, sq2, sv1, sv2, sb1, sb2, fd_k.val);

        // 2. K_gram = K @ K^T [64, 64]
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            CHUNK_B, CHUNK_B, SV,
            &one,
            ws.k_pack, SV, (long long)CHUNK_B * SV,
            ws.k_pack, SV, (long long)CHUNK_B * SV,
            &zero,
            ws.gram_mat, CHUNK_B, (long long)CHUNK_B * CHUNK_B,
            n_batch));

        // 3. A = (I + L)^{-1} in place in gram_mat
        wy_invert_l_kernel<<<n_batch, CHUNK_B, 0, stream>>>(ws.gram_mat, ws.g_cum, ws.beta_buf);

        // 4. W = A @ K_beta_g (into first 64 rows of wq_hat -> stride 128*SV)
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            SV, CHUNK_B, CHUNK_B,
            &one,
            ws.k_beta_g, SV, (long long)CHUNK_B * SV,
            ws.gram_mat, CHUNK_B, (long long)CHUNK_B * CHUNK_B,
            &zero,
            ws.wq_hat, SV, (long long)128 * SV,
            n_batch));

        // 5. U = A @ V_beta (overwrite v_beta? Keep separate delta buffer -> write into v_beta is safe too, wait we can write U into v_beta!)
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            SV, CHUNK_B, CHUNK_B,
            &one,
            ws.v_beta, SV, (long long)CHUNK_B * SV,
            ws.gram_mat, CHUNK_B, (long long)CHUNK_B * CHUNK_B,
            &zero,
            ws.k_beta_g, SV, (long long)CHUNK_B * SV, // k_beta_g is free now -> holds U!
            n_batch));

        // 6. Q_gram = Q @ K^T -> mask into M (reuse gram_mat)
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            CHUNK_B, CHUNK_B, SV,
            &one,
            ws.k_pack, SV, (long long)CHUNK_B * SV,
            ws.q_pack, SV, (long long)CHUNK_B * SV,
            &zero,
            ws.gram_mat, CHUNK_B, (long long)CHUNK_B * CHUNK_B,
            n_batch));
        wy_mask_qgram_kernel<<<n_batch, 256, 0, stream>>>(ws.gram_mat, ws.g_cum);

        // 7. Sequential chunk loop over c = 0..n_chunks-1 (H=32 heads batched)
        for (int c = 0; c < n_chunks; ++c) {
            const size_t tile128_c = (size_t)c * H * 128 * SV;
            const size_t tile64_c  = (size_t)c * H * CHUNK_B * SV;
            const float * wq_c     = ws.wq_hat + tile128_c;
            float * wso_c          = ws.ws_o + tile128_c;
            const float * u_c      = ws.k_beta_g + tile64_c; // holds U
            float * del_c          = ws.delta + tile64_c;
            const float * kt_c     = ws.k_tilde + tile64_c;

            // [WS_c; O_inter_c] = [W_c; Q_hat_c] @ S_work
            CHECK_CUBLAS(cublasSgemmStridedBatched(
                cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                SV, 128, SV,
                &one,
                ws.s_work, SV, (long long)SV * SV,
                wq_c, SV, (long long)128 * SV,
                &zero,
                wso_c, SV, (long long)128 * SV,
                H));

            // Delta_c = U_c - WS_c; S_work *= chunk_dec[c*H + h]
            wy_chunk_epilogue_kernel<<<dim3(H, 64), 256, 0, stream>>>(
                u_c, wso_c, ws.chunk_dec + (size_t)c * H, del_c, ws.s_work, H);

            // S_work += K_tilde_c^T @ Delta_c
            CHECK_CUBLAS(cublasSgemmStridedBatched(
                cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                SV, SV, CHUNK_B,
                &one,
                kt_c, SV, (long long)CHUNK_B * SV,
                del_c, SV, (long long)CHUNK_B * SV,
                &one,
                ws.s_work, SV, (long long)SV * SV,
                H));
        }

        // 8. Post-chunk O_inter += M @ Delta across all n_batch tiles
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            SV, CHUNK_B, CHUNK_B,
            &one,
            ws.delta, SV, (long long)CHUNK_B * SV,
            ws.gram_mat, CHUNK_B, (long long)CHUNK_B * CHUNK_B,
            &one,
            ws.ws_o + CHUNK_B * SV, SV, (long long)128 * SV,
            n_batch));

        // 9. Unpack O back into dst_d[0..t_wy-1]
        wy_unpack_out_kernel<<<dim3(H, n_chunks), 128, 0, stream>>>(ws.ws_o, dst_d, H, scale);
    }

    const int tail_tokens = n_tokens - t_wy;
    const float * tail_s_in = (n_chunks > 0) ? ws.s_work : s_in_d;
    const float * tail_q    = q_d + (size_t)t_wy * sq2;
    const float * tail_k    = k_d + (size_t)t_wy * sq2;
    const float * tail_v    = v_d + (size_t)t_wy * sv2;
    const float * tail_g    = g_d + (size_t)t_wy * sb2;
    const float * tail_b    = b_d + (size_t)t_wy * sb2;
    float * tail_dst        = dst_d + (size_t)t_wy * H * SV;

    const int64_t slot_stride = (int64_t)H * SV * SV;
    dim3 grid(H, 1, SV / 16); // num_warps=4, cols_per_warp=4 -> 16 cols/CTA -> 8 CTAs/head = 256 CTAs >= 24 SMs
    dim3 block(32, 4, 1);
    if (K > 1) {
        stock_gdn_kernel<true, 4><<<grid, block, 0, stream>>>(
            tail_q, tail_k, tail_v, tail_g, tail_b, tail_s_in,
            tail_dst, state_out_d, H, tail_tokens, 1,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            fd_k.val, fd_1.val, scale, slot_stride, K);
    } else {
        stock_gdn_kernel<false, 4><<<grid, block, 0, stream>>>(
            tail_q, tail_k, tail_v, tail_g, tail_b, tail_s_in,
            tail_dst, state_out_d, H, tail_tokens, 1,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            fd_k.val, fd_1.val, scale, slot_stride, K);
    }
    CHECK_CUDA(cudaGetLastError());
}

static void launch_stock(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_in_d,
        float * dst_d, float * state_out_d,
        int H, int H_k, int n_tokens, int K,
        cudaStream_t stream) {
    const int64_t sq1 = SV, sq2 = SV * H_k, sq3 = SV * H_k * n_tokens;
    const int64_t sv1 = SV, sv2 = SV * H,   sv3 = SV * H * n_tokens;
    const int64_t sb1 = 1,  sb2 = H,        sb3 = H * n_tokens;
    const FastDiv fd_k = make_fastdiv(H_k);
    const FastDiv fd_1 = make_fastdiv(1);
    const float scale = 1.0f / sqrtf((float)SV);
    const int64_t slot_stride = (int64_t)H * SV * SV;
    dim3 grid(H, 1, SV / 16);
    dim3 block(32, 4, 1);
    if (K > 1) {
        stock_gdn_kernel<true, 4><<<grid, block, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d, s_in_d, dst_d, state_out_d,
            H, n_tokens, 1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            fd_k.val, fd_1.val, scale, slot_stride, K);
    } else {
        stock_gdn_kernel<false, 4><<<grid, block, 0, stream>>>(
            q_d, k_d, v_d, g_d, b_d, s_in_d, dst_d, state_out_d,
            H, n_tokens, 1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            fd_k.val, fd_1.val, scale, slot_stride, K);
    }
    CHECK_CUDA(cudaGetLastError());
}

// --- Exact FP64 CPU reference matching gated_delta_net_cuda ---
static void fp64_reference(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta,
        const std::vector<float> & s_in,
        std::vector<double> & dst_out,
        std::vector<double> & state_out,
        int H, int H_k, int n_tokens, int K) {
    const double scale = 1.0 / std::sqrt((double)SV);
    dst_out.assign((size_t)n_tokens * H * SV, 0.0);
    state_out.assign((size_t)K * H * SV * SV, 0.0);

    for (int h = 0; h < H; ++h) {
        const int iq1 = h % H_k;
        std::vector<double> S((size_t)SV * SV);
        const size_t h_off = (size_t)h * SV * SV;
        for (int idx = 0; idx < SV * SV; ++idx) S[idx] = (double)s_in[h_off + idx];

        for (int t = 0; t < n_tokens; ++t) {
            const float * qt = q.data() + ((size_t)t * H_k + iq1) * SV;
            const float * kt = k.data() + ((size_t)t * H_k + iq1) * SV;
            const float * vt = v.data() + ((size_t)t * H + h) * SV;
            const size_t gb = (size_t)t * H + h;
            const double gv = std::exp((double)g[gb]);
            const double bv = (double)beta[gb];

            double * out_t = dst_out.data() + ((size_t)t * H + h) * SV;
            for (int col = 0; col < SV; ++col) {
                double kv = 0.0;
                for (int i = 0; i < SV; ++i) kv += S[(size_t)col * SV + i] * (double)kt[i];
                double delta = ((double)vt[col] - gv * kv) * bv;
                double attn  = 0.0;
                for (int i = 0; i < SV; ++i) {
                    double snew = gv * S[(size_t)col * SV + i] + (double)kt[i] * delta;
                    S[(size_t)col * SV + i] = snew;
                    attn += snew * (double)qt[i];
                }
                out_t[col] = attn * scale;
            }

            if (K > 1) {
                const int target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    double * sl = state_out.data() + (size_t)target_slot * H * SV * SV + h_off;
                    std::copy(S.begin(), S.end(), sl);
                }
            }
        }
        if (K == 1) {
            double * sl = state_out.data() + h_off;
            std::copy(S.begin(), S.end(), sl);
        }
    }
}

struct ErrStats {
    double max_abs = 0.0;
    double rel_rmse = 0.0;
    int nonfinite = 0;
};

static ErrStats compare_fp64(const std::vector<float> & got, const std::vector<double> & ref) {
    ErrStats st;
    double sq_err = 0.0, sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
            st.nonfinite++;
            continue;
        }
        double e = std::abs((double)got[i] - ref[i]);
        if (e > st.max_abs) st.max_abs = e;
        sq_err += e * e;
        sq_ref += ref[i] * ref[i];
    }
    st.rel_rmse = (sq_ref > 0.0) ? std::sqrt(sq_err / sq_ref) : std::sqrt(sq_err);
    return st;
}

int main(int argc, char ** argv) {
    bool check_only = (argc > 1 && std::strcmp(argv[1], "--check-only") == 0);
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (cc %d.%d, SMs=%d)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    cublasHandle_t cublas{};
    CHECK_CUBLAS(cublasCreate(&cublas));
    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(cublas, stream));

    const int max_T = 2048;
    const int max_K = 2;
    const int max_chunks = max_T / CHUNK_B;
    WyWorkspace ws;
    ws.init(NH, max_chunks);
    printf("WyWorkspace bytes (max_T=%d): %.2f MiB\n", max_T, ws.bytes / (1024.0 * 1024.0));

    float *d_q, *d_k, *d_v, *d_g, *d_b, *d_sin, *d_dst, *d_sout;
    CHECK_CUDA(cudaMalloc(&d_q,   (size_t)max_T * HK * SV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k,   (size_t)max_T * HK * SV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v,   (size_t)max_T * NH * SV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g,   (size_t)max_T * NH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b,   (size_t)max_T * NH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sin, (size_t)NH * SV * SV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, (size_t)max_T * NH * SV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sout,(size_t)max_K * NH * SV * SV * sizeof(float)));

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    struct NumCase {
        int T;
        int K;
        float g_lo, g_hi;
        float b_lo, b_hi;
        const char * label;
    };
    std::vector<NumCase> num_cases;
    for (int K : {1, 2}) {
        for (int T : {124, 128, 512, 1024, 2048}) {
            num_cases.push_back({T, K, -1.5f, -0.005f, 0.05f, 0.95f, "norm"});
        }
    }
    num_cases.push_back({512, 2, -5.0f, -0.0001f, 0.01f, 0.99f, "harsh"});

    bool all_ok = true;
    for (const auto & nc : num_cases) {
        const int T = nc.T;
        const int K = nc.K;
        std::uniform_real_distribution<float> ug(nc.g_lo, nc.g_hi);
        std::uniform_real_distribution<float> ub(nc.b_lo, nc.b_hi);

        std::vector<float> h_q((size_t)T * HK * SV), h_k((size_t)T * HK * SV);
        std::vector<float> h_v((size_t)T * NH * SV), h_g((size_t)T * NH), h_b((size_t)T * NH);
        std::vector<float> h_sin((size_t)NH * SV * SV);

        for (int row = 0; row < T * HK; ++row) {
            double nq = 0.0, nk = 0.0;
            for (int i = 0; i < SV; ++i) {
                float qv = nd(rng), kv = nd(rng);
                h_q[(size_t)row * SV + i] = qv;
                h_k[(size_t)row * SV + i] = kv;
                nq += (double)qv * qv;
                nk += (double)kv * kv;
            }
            float iq = 1.0f / (float)std::sqrt(nq + 1e-12);
            float ik = 1.0f / (float)std::sqrt(nk + 1e-12);
            for (int i = 0; i < SV; ++i) {
                h_q[(size_t)row * SV + i] *= iq;
                h_k[(size_t)row * SV + i] *= ik;
            }
        }
        for (float & x : h_v)   x = nd(rng) * 0.5f;
        for (float & x : h_g)   x = ug(rng);
        for (float & x : h_b)   x = ub(rng);
        for (float & x : h_sin) x = nd(rng) * 0.1f;

        CHECK_CUDA(cudaMemcpyAsync(d_q,   h_q.data(),   h_q.size() * sizeof(float),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_k,   h_k.data(),   h_k.size() * sizeof(float),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_v,   h_v.data(),   h_v.size() * sizeof(float),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_g,   h_g.data(),   h_g.size() * sizeof(float),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_b,   h_b.data(),   h_b.size() * sizeof(float),   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_sin, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        std::vector<double> ref_dst, ref_st;
        fp64_reference(h_q, h_k, h_v, h_g, h_b, h_sin, ref_dst, ref_st, NH, HK, T, K);

        std::vector<float> got_dst(ref_dst.size()), got_st(ref_st.size());

        // Run Stock
        CHECK_CUDA(cudaMemsetAsync(d_dst,  0, got_dst.size() * sizeof(float), stream));
        CHECK_CUDA(cudaMemsetAsync(d_sout, 0, got_st.size()  * sizeof(float), stream));
        launch_stock(d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, K, stream);
        CHECK_CUDA(cudaMemcpyAsync(got_dst.data(), d_dst,  got_dst.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(got_st.data(),  d_sout, got_st.size()  * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        ErrStats st_dst = compare_fp64(got_dst, ref_dst);
        ErrStats st_st  = compare_fp64(got_st,  ref_st);

        // Run Chunked WY
        CHECK_CUDA(cudaMemsetAsync(d_dst,  0, got_dst.size() * sizeof(float), stream));
        CHECK_CUDA(cudaMemsetAsync(d_sout, 0, got_st.size()  * sizeof(float), stream));
        launch_chunked_wy(cublas, ws, d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, K, stream);
        CHECK_CUDA(cudaMemcpyAsync(got_dst.data(), d_dst,  got_dst.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(got_st.data(),  d_sout, got_st.size()  * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        ErrStats wy_dst = compare_fp64(got_dst, ref_dst);
        ErrStats wy_st  = compare_fp64(got_st,  ref_st);

        double dst_ratio = wy_dst.rel_rmse / std::max(st_dst.rel_rmse, 1e-7);
        double st_ratio  = wy_st.rel_rmse  / std::max(st_st.rel_rmse,  1e-7);
        bool pass = (wy_dst.nonfinite == 0 && wy_st.nonfinite == 0 &&
                     wy_dst.rel_rmse <= 1e-4 && wy_st.rel_rmse <= 1e-4 &&
                     dst_ratio <= 2.0 && st_ratio <= 2.0);
        if (!pass) all_ok = false;

        printf("NUM[%s] T=%4d K=%d | stock dst=%.2e st=%.2e | wy dst=%.2e st=%.2e | ratio dst=%.2fx st=%.2fx | nonfin=%d+%d -> %s\n",
               nc.label, T, K, st_dst.rel_rmse, st_st.rel_rmse,
               wy_dst.rel_rmse, wy_st.rel_rmse, dst_ratio, st_ratio,
               wy_dst.nonfinite, wy_st.nonfinite, pass ? "PASS" : "FAIL");
    }

    if (!all_ok) {
        fprintf(stderr, "Numerical check FAILED\n");
        return 4;
    }
    if (check_only) return 0;

    // ABBA Timing (64 warmup + 100 timed per arm, K=2)
    auto time_arm = [&](bool use_wy, int T, int reps) -> double {
        cudaEvent_t ev0, ev1;
        CHECK_CUDA(cudaEventCreate(&ev0));
        CHECK_CUDA(cudaEventCreate(&ev1));
        for (int i = 0; i < 16; ++i) {
            if (use_wy) launch_chunked_wy(cublas, ws, d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, 2, stream);
            else        launch_stock(d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, 2, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaEventRecord(ev0, stream));
        for (int i = 0; i < reps; ++i) {
            if (use_wy) launch_chunked_wy(cublas, ws, d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, 2, stream);
            else        launch_stock(d_q, d_k, d_v, d_g, d_b, d_sin, d_dst, d_sout, NH, HK, T, 2, stream);
        }
        CHECK_CUDA(cudaEventRecord(ev1, stream));
        CHECK_CUDA(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
        CHECK_CUDA(cudaEventDestroy(ev0));
        CHECK_CUDA(cudaEventDestroy(ev1));
        return (double)ms / reps;
    };

    const int timing_Ts[] = {1024, 2048};
    for (int T : timing_Ts) {
        const int reps = 50;
        double a1 = time_arm(false, T, reps);
        double b1 = time_arm(true,  T, reps);
        double b2 = time_arm(true,  T, reps);
        double a2 = time_arm(false, T, reps);
        double stock_avg = 0.5 * (a1 + a2);
        double wy_avg    = 0.5 * (b1 + b2);
        double speedup   = stock_avg / wy_avg;
        printf("TIMING ABBA T=%4d K=2 | stock=[%.4f, %.4f] avg=%.4f ms | wy=[%.4f, %.4f] avg=%.4f ms | speedup=%.3fx (save=%.4f ms/call, 240calls=%.1f ms)\n",
               T, a1, a2, stock_avg, b1, b2, wy_avg, speedup, stock_avg - wy_avg, (stock_avg - wy_avg) * 240.0);
    }

    ws.destroy();
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_b);
    cudaFree(d_sin); cudaFree(d_dst); cudaFree(d_sout);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    return 0;
}
