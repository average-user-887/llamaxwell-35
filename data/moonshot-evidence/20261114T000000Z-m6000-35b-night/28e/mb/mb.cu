// Card 28e microbench: Q4_0 MoE expert matmul at prefill shapes, FP32 dequant-in-smem SGEMM tile
// (candidate) vs FP64 reference and a q8_1-activation emulation (what int8 mmq computes).
// Shapes: T=2048 tokens/ubatch, 8 of 256 experts, gate/up rows 512 K 2048 (broadcast input), down rows 2048 K 512.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

static constexpr int BM = 128;

// BN tokens x BM=128 weight rows per CTA; NT = 16*BN/TN threads, each 8 rows x TN tokens.
// BK=32: one Q4_0 block per k-tile; BK=16: half a block, k permuted identically for A and B.
template <int BN, int TN, int BK, int MINB, bool BCAST>
__global__ void __launch_bounds__(16 * BN / TN, MINB)
moe_f32_q40(const uint8_t * __restrict__ W, const float * __restrict__ X,
            const int * __restrict__ ids_src, const int * __restrict__ ids_dst,
            const int * __restrict__ bounds, const int2 * __restrict__ tiles,
            float * __restrict__ Y, const int rows, const int K) {
    constexpr int NT  = 16 * BN / TN;
    constexpr int AIT = 256 / NT;            // (row, part) items per thread
    constexpr int NUS = BK / 8;              // ushorts of qs per part
    constexpr int BIT = BN * BK / 4 / NT;    // float4 of B per thread
    __shared__ __align__(16) float As[2][BK][BM];
    __shared__ __align__(16) float Bs[2][BK][BN];

    if ((int) blockIdx.y >= tiles[0].x) return;
    const int2 tl = tiles[1 + blockIdx.y];
    const int e = tl.x;
    const int c0 = bounds[e] + tl.y * BN;
    const int ncols = min(BN, bounds[e + 1] - c0);
    const int r0 = blockIdx.x * BM;
    const int tid = threadIdx.x;
    const int mycol = tid % BN;
    const int s = mycol < ncols ? ids_src[c0 + mycol] : -1;
    const float * xs = X + (size_t) (s < 0 ? 0 : s) * K;

    const int nkt = K / BK;
    const size_t rowstride = (size_t) (K / 32) * 18;
    const uint8_t * Wb = W + ((size_t) e * rows + r0) * rowstride;

    unsigned short ad[AIT];
    unsigned short aq[AIT][NUS];
    float4 bv[BIT];

    auto loadA = [&](int kt) {
        const int kb = BK == 32 ? kt : kt >> 1, h = BK == 32 ? 0 : kt & 1;
#pragma unroll
        for (int i = 0; i < AIT; ++i) {
            const int it = tid + i * NT, row = it & (BM - 1), part = it >> 7;
            const unsigned short * p = (const unsigned short *) (Wb + row * rowstride + kb * 18);
            ad[i] = __ldg(p);
#pragma unroll
            for (int j = 0; j < NUS; ++j) aq[i][j] = __ldg(p + 1 + h * 4 + part * NUS + j);
        }
    };
    auto storeA = [&](int buf) {
#pragma unroll
        for (int i = 0; i < AIT; ++i) {
            const int it = tid + i * NT, row = it & (BM - 1), part = it >> 7;
            const float d = __half2float(__ushort_as_half(ad[i]));
#pragma unroll
            for (int j = 0; j < NUS; ++j) {
#pragma unroll
                for (int bb = 0; bb < 2; ++bb) {
                    const unsigned byte = (aq[i][j] >> (8 * bb)) & 0xFF;
                    const int idx = part * (BK / 4) + 2 * j + bb;
                    const float lo = __int_as_float(0x4B000000 | (byte & 0xF)) - 8388616.0f;
                    const float hi = __int_as_float(0x4B000000 | (byte >> 4)) - 8388616.0f;
                    As[buf][idx][row]          = lo * d;
                    As[buf][idx + BK / 2][row] = hi * d;
                }
            }
        }
    };
    auto loadB = [&](int kt) {
        const int kb = BK == 32 ? kt : kt >> 1, h = BK == 32 ? 0 : kt & 1;
#pragma unroll
        for (int i = 0; i < BIT; ++i) {
            const int f = tid + i * NT, kq4 = (f / BN) * 4;
            const int off = BK == 32 ? kq4 : (kq4 < 8 ? h * 8 + kq4 : 16 + h * 8 + kq4 - 8);
            bv[i] = s >= 0 ? __ldg((const float4 *) (xs + kb * 32 + off)) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto storeB = [&](int buf) {
#pragma unroll
        for (int i = 0; i < BIT; ++i) {
            const int f = tid + i * NT, kq4 = (f / BN) * 4;
            Bs[buf][kq4 + 0][mycol] = bv[i].x;
            Bs[buf][kq4 + 1][mycol] = bv[i].y;
            Bs[buf][kq4 + 2][mycol] = bv[i].z;
            Bs[buf][kq4 + 3][mycol] = bv[i].w;
        }
    };

    const int tx = (tid & 7) | (((tid >> 5) & 1) << 3), ty = ((tid >> 3) & 3) | ((tid >> 6) << 2);  // warp = 8x4 thread block: A 128 B + B 64 B per LDS.128
    float acc[8][TN];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    loadA(0); loadB(0); storeA(0); storeB(0);
    __syncthreads();
    for (int kt = 0; kt < nkt; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nkt) { loadA(kt + 1); loadB(kt + 1); }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a0 = *(const float4 *) &As[buf][k][tx * 4];
            const float4 a1 = *(const float4 *) &As[buf][k][64 + tx * 4];
            float b[TN];
            const float4 b0 = *(const float4 *) &Bs[buf][k][ty * 4];
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            if constexpr (TN == 8) {
                const float4 b1 = *(const float4 *) &Bs[buf][k][BN / 2 + ty * 4];
                b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            }
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (kt + 1 < nkt) { storeA(buf ^ 1); storeB(buf ^ 1); }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int col = TN == 4 ? ty * 4 + j : (j < 4 ? ty * 4 + j : BN / 2 + ty * 4 + j - 4);
        if (col < ncols) {
            float * y = Y + (size_t) ids_dst[c0 + col] * rows + r0;
            *(float4 *) (y + tx * 4)      = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            *(float4 *) (y + 64 + tx * 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
        }
    }
}

struct Routing { std::vector<int> bounds, ids_src_b, ids_src_nb, ids_dst; std::vector<int2> tiles, t64, t32, t16; int ntiles; int maxc; };

static Routing make_routing(int T, int NU, int E, double sigma, std::mt19937 & rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(1e-12, 1.0);
    std::vector<double> lw(E);
    for (auto & v : lw) v = sigma * nd(rng);
    std::vector<std::vector<int>> per(E);
    std::vector<std::pair<double, int>> sc(E);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < E; ++e) sc[e] = {lw[e] - std::log(-std::log(ud(rng))), e};
        std::partial_sort(sc.begin(), sc.begin() + NU, sc.end(), [](auto & a, auto & b) { return a.first > b.first; });
        for (int u = 0; u < NU; ++u) per[sc[u].second].push_back(t * NU + u);
    }
    Routing r; r.bounds.push_back(0); r.maxc = 0;
    for (int e = 0; e < E; ++e) {
        for (int s : per[e]) { r.ids_dst.push_back(s); r.ids_src_b.push_back(s / NU); r.ids_src_nb.push_back(s); }
        r.bounds.push_back((int) r.ids_dst.size());
        r.maxc = std::max(r.maxc, (int) per[e].size());
    }
    r.tiles.push_back(make_int2(0, 0));
    for (int e = 0; e < E; ++e) {
        const int n = (int) per[e].size();
        for (int j = 0; j * 64 < n; ++j) r.tiles.push_back(make_int2(e, j));
    }
    r.ntiles = (int) r.tiles.size() - 1;
    r.tiles[0].x = r.ntiles;
    r.t64.push_back(make_int2(0, 0)); r.t32.push_back(make_int2(0, 0)); r.t16.push_back(make_int2(0, 0));
    for (int e = 0; e < E; ++e) {  // same split as ggml_cuda_mmq_make_tail_tiles
        const int n = (int) per[e].size(), full = n / 64, rem = n - full * 64;
        for (int j = 0; j < full + (rem > 32); ++j) r.t64.push_back(make_int2(e, j));
        if (rem > 16 && rem <= 32) r.t32.push_back(make_int2(e, full * 2));
        if (rem > 0 && rem <= 16) r.t16.push_back(make_int2(e, full * 4));
    }
    r.t64[0].x = (int) r.t64.size() - 1; r.t32[0].x = (int) r.t32.size() - 1; r.t16[0].x = (int) r.t16.size() - 1;
    return r;
}

static float h2f(uint16_t h) { __half_raw r; r.x = h; return __half2float(__half(r)); }
static uint16_t f2h(float f) { __half h = __float2half_rn(f); return ((__half_raw) h).x; }

int main(int argc, char ** argv) {
    const int T = 2048, NU = 8, E = 256, NS = T * NU;
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    std::mt19937 rng(28);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sm_%d%d nsm %d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    for (int shape = 0; shape < 2; ++shape) {
        const bool bcast = shape == 0;
        const int rows = bcast ? 512 : 2048, K = bcast ? 2048 : 512, nkb = K / 32;
        const size_t wbytes = (size_t) E * rows * nkb * 18;
        std::vector<uint8_t> W(wbytes);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (size_t b = 0; b < (size_t) E * rows * nkb; ++b) {
            uint8_t * p = &W[b * 18];
            uint16_t d = f2h(0.004f + 0.002f * std::fabs(nd(rng)));
            memcpy(p, &d, 2);
            for (int j = 0; j < 16; ++j) {
                // roughly Gaussian nibbles centred at 8
                int lo = std::min(15, std::max(0, (int) std::lround(8 + 3 * nd(rng))));
                int hi = std::min(15, std::max(0, (int) std::lround(8 + 3 * nd(rng))));
                p[2 + j] = (uint8_t) (lo | (hi << 4));
            }
        }
        const size_t xrows = bcast ? T : NS;
        std::vector<float> X(xrows * K);
        for (auto & v : X) v = bcast ? nd(rng) : nd(rng) * std::fabs(nd(rng)) * 0.3f;  // down input heavier-tailed (silu*up)

        uint8_t * dW; float * dX, * dY; int * dsrc, * ddst, * dbnd; int2 * dtiles, * dt64, * dt32, * dt16;
        CK(cudaMalloc(&dW, wbytes)); CK(cudaMalloc(&dX, X.size() * 4)); CK(cudaMalloc(&dY, (size_t) NS * rows * 4));
        CK(cudaMemcpy(dW, W.data(), wbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dsrc, NS * 4)); CK(cudaMalloc(&ddst, NS * 4)); CK(cudaMalloc(&dbnd, (E + 1) * 4));
        CK(cudaMalloc(&dtiles, (NS / 64 + E + 1) * sizeof(int2)));
        CK(cudaMalloc(&dt64, (NS / 64 + E + 1) * sizeof(int2))); CK(cudaMalloc(&dt32, (E + 1) * sizeof(int2))); CK(cudaMalloc(&dt16, (E + 1) * sizeof(int2)));

        for (int rt = 0; rt < 2; ++rt) {
            const double sigma = rt == 0 ? 0.0 : 0.8;
            Routing R = make_routing(T, NU, E, sigma, rng);
            CK(cudaMemcpy(dsrc, (bcast ? R.ids_src_b : R.ids_src_nb).data(), NS * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ddst, R.ids_dst.data(), NS * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dbnd, R.bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dtiles, R.tiles.data(), R.tiles.size() * sizeof(int2), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dt64, R.t64.data(), R.t64.size() * sizeof(int2), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dt32, R.t32.data(), R.t32.size() * sizeof(int2), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dt16, R.t16.data(), R.t16.size() * sizeof(int2), cudaMemcpyHostToDevice));
            const int n64 = R.t64[0].x, n32 = R.t32[0].x, n16 = R.t16[0].x;
            
            const double flop = 2.0 * rows * K * NS;

            static const char * vname[] = {"J64only-bk32", "tail-bk32-mb2", "tail-bk16-mb2", "tail-bk16-mb3", "tail-bk32-mb3", "t8-bk16-mb2", "t8-bk16-mb1", "t8-bk32-mb1"};
            for (int var = 0; var < 8; ++var) {
                CK(cudaMemset(dY, 0, (size_t) NS * rows * 4));
                const int gx = rows / BM;
#define L3(BK_, MB_) do { \
                    moe_f32_q40<64, 4, BK_, MB_, false><<<dim3(gx, n64), 256>>>(dW, dX, dsrc, ddst, dbnd, dt64, dY, rows, K); \
                    if (n32) moe_f32_q40<32, 4, BK_, MB_, false><<<dim3(gx, n32), 128>>>(dW, dX, dsrc, ddst, dbnd, dt32, dY, rows, K); \
                    if (n16) moe_f32_q40<16, 4, BK_, MB_, false><<<dim3(gx, n16), 64>>>(dW, dX, dsrc, ddst, dbnd, dt16, dY, rows, K); } while (0)
#define L8(BK_, MB_) do { \
                    moe_f32_q40<64, 8, BK_, MB_, false><<<dim3(gx, n64), 128>>>(dW, dX, dsrc, ddst, dbnd, dt64, dY, rows, K); \
                    if (n32) moe_f32_q40<32, 8, BK_, MB_, false><<<dim3(gx, n32), 64>>>(dW, dX, dsrc, ddst, dbnd, dt32, dY, rows, K); \
                    if (n16) moe_f32_q40<16, 4, BK_, MB_, false><<<dim3(gx, n16), 64>>>(dW, dX, dsrc, ddst, dbnd, dt16, dY, rows, K); } while (0)
                auto launch = [&]() {
                    switch (var) {
                        case 0: moe_f32_q40<64, 4, 32, 2, false><<<dim3(gx, R.ntiles), 256>>>(dW, dX, dsrc, ddst, dbnd, dtiles, dY, rows, K); break;
                        case 1: L3(32, 2); break;
                        case 2: L3(16, 2); break;
                        case 3: L3(16, 3); break;
                        case 4: L3(32, 3); break;
                        case 5: L8(16, 2); break;
                        case 6: L8(16, 1); break;
                        case 7: L8(32, 1); break;
                    }
                };
                for (int i = 0; i < 3; ++i) launch();
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                std::vector<float> ts;
                for (int i = 0; i < reps; ++i) {
                    cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
                    float ms; cudaEventElapsedTime(&ms, a, b); ts.push_back(ms);
                }
                std::sort(ts.begin(), ts.end());
                const float med = ts[ts.size() / 2];
                std::vector<float> Y((size_t) NS * rows);
                CK(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost));

                // accuracy on sampled outputs: FP64 reference, q8_1 emulation, candidate
                std::uniform_int_distribution<int> us(0, NS - 1), ur(0, rows - 1);
                std::vector<int> e_of(NS);
                for (int e = 0; e < E; ++e) for (int i = R.bounds[e]; i < R.bounds[e + 1]; ++i) e_of[R.ids_dst[i]] = e;
                double se_c = 0, se_q = 0, sr = 0, mx_c = 0; int nz = 0;
                const int NSAMP = 6000;
                for (int sIdx = 0; sIdx < NSAMP; ++sIdx) {
                    const int s = us(rng), r = ur(rng), e = e_of[s];
                    const float * x = &X[(size_t) (bcast ? s / NU : s) * K];
                    const uint8_t * wr = &W[((size_t) e * rows + r) * nkb * 18];
                    double ref = 0, q8 = 0;
                    for (int kb = 0; kb < nkb; ++kb) {
                        uint16_t dh; memcpy(&dh, wr + kb * 18, 2); const double dw = h2f(dh);
                        const float * xb = x + kb * 32;
                        float amax = 0; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
                        const float d8 = amax / 127.f; int qx[32]; int sumq = 0;
                        for (int j = 0; j < 32; ++j) { qx[j] = amax == 0 ? 0 : (int) std::lround(xb[j] / d8); sumq += qx[j]; }
                        const double d8h = h2f(f2h(d8)), s8h = h2f(f2h(d8 * sumq));
                        long sumi = 0;
                        for (int j = 0; j < 16; ++j) {
                            const int lo = wr[kb * 18 + 2 + j] & 0xF, hi = wr[kb * 18 + 2 + j] >> 4;
                            ref += dw * (lo - 8) * (double) xb[j] + dw * (hi - 8) * (double) xb[j + 16];
                            sumi += (long) lo * qx[j] + (long) hi * qx[j + 16];
                        }
                        q8 += dw * (d8h * sumi - 8.0 * s8h);
                    }
                    const double c = Y[(size_t) s * rows + r];
                    se_c += (c - ref) * (c - ref); se_q += (q8 - ref) * (q8 - ref); sr += ref * ref;
                    mx_c = std::max(mx_c, std::fabs(c - ref)); nz += c == 0.0;
                }
                printf("%s rows %d K %d routing %s (tiles %d, max/expert %d) var %s: median %.3f ms min %.3f  %.2f TFLOPS | relRMSE vs FP64: cand %.3e  q8_1-emul %.3e  cand max abs %.3e zeros %d\n",
                    bcast ? "gate/up" : "down", rows, K, rt == 0 ? "uniform" : "skew0.8", R.ntiles, R.maxc,
                    vname[var], med, ts[0], flop / med * 1e-9,
                    std::sqrt(se_c / sr), std::sqrt(se_q / sr), mx_c, nz);
            }
        }
        cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dsrc); cudaFree(ddst); cudaFree(dbnd); cudaFree(dtiles); cudaFree(dt64); cudaFree(dt32); cudaFree(dt16);
    }
    return 0;
}
