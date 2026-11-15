// Card 28: QK GEMM layout microbench. scores[key, col] = scale * K . Q for one KV head group.
// TN: K row-major [key][256] (ld 256), OP_T.  NN: K transposed [d][ldk] (ld ldk), OP_N.
// Checks NN vs TN and both vs FP64 on sampled entries.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
int main() {
    const int D = 256, NC = 512;
    int keys_list[] = {30720, 61440, 92160};
    cublasHandle_t h; cublasCreate(&h);
    for (int keys : keys_list) {
        const int ldk = (keys + 31) / 32 * 32;
        std::vector<float> K(size_t(keys) * D), Q(size_t(NC) * D), KT(size_t(ldk) * D, 0.f);
        std::mt19937 rng(1); std::normal_distribution<float> nd(0, 1);
        for (auto & x : K) x = nd(rng);
        for (auto & x : Q) x = nd(rng);
        for (int k = 0; k < keys; ++k) for (int d = 0; d < D; ++d) KT[size_t(d) * ldk + k] = K[size_t(k) * D + d];
        float *dK, *dKT, *dQ, *dS1, *dS2;
        cudaMalloc(&dK, K.size() * 4); cudaMalloc(&dKT, KT.size() * 4); cudaMalloc(&dQ, Q.size() * 4);
        cudaMalloc(&dS1, size_t(keys) * NC * 4); cudaMalloc(&dS2, size_t(keys) * NC * 4);
        cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dKT, KT.data(), KT.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dQ, Q.data(), Q.size() * 4, cudaMemcpyHostToDevice);
        const float sc = 0.0625f, z = 0.f;
        auto tn = [&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, NC, D, &sc, dK, D, dQ, D, &z, dS1, keys); };
        auto nn = [&] { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, keys, NC, D, &sc, dKT, ldk, dQ, D, &z, dS2, keys); };
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        auto tim = [&](auto f) { f(); cudaEventRecord(a); for (int i = 0; i < 10; ++i) f(); cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); return cudaGetLastError() == cudaSuccess ? ms / 10 : -1.f; };
        const double fl0 = 2.0 * keys * NC * D;
        // strided batch over 4096-key splits (PV-like M)
        const int sp = 4096, nb = keys / sp;
        printf("keys=%d TN-batched4096 %.3f ms NN-batched4096 %.3f ms\n", keys,
            tim([&] { cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, sp, NC, D, &sc, dK, D, size_t(sp) * D, dQ, D, 0, &z, dS1, keys, sp, nb); }),
            tim([&] { cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, sp, NC, D, &sc, dKT, ldk, sp, dQ, D, 0, &z, dS2, keys, sp, nb); }));
        // transposed output: S^T[col, key] = Q^T K  (OP_T Q, OP_N K)
        printf("keys=%d ST(TN, M=512) %.3f ms\n", keys, tim([&] { cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, NC, keys, D, &sc, dQ, D, dK, D, &z, dS2, NC); }));
        for (int algo = 0; algo < 24; ++algo) {
            float t1 = tim([&] { if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, NC, D, &sc, dK, CUDA_R_32F, D, dQ, CUDA_R_32F, D, &z, dS1, CUDA_R_32F, keys, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) algo) != CUBLAS_STATUS_SUCCESS) cudaEventRecord(b); });
            float t2 = tim([&] { cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, keys, NC, D, &sc, dKT, CUDA_R_32F, ldk, dQ, CUDA_R_32F, D, &z, dS2, CUDA_R_32F, keys, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) algo); });
            printf("keys=%d algo %d TN %.3f (%.2f TF) NN %.3f (%.2f TF)\n", keys, algo, t1, fl0 / t1 / 1e9, t2, fl0 / t2 / 1e9);
        }
        float t[2] = {0, 0};
        for (int rep = 0; rep < 3; ++rep) for (int v = 0; v < 2; ++v) {
            if (v) nn(); else tn(); cudaEventRecord(a);
            for (int i = 0; i < 20; ++i) if (v) nn(); else tn();
            cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (rep) t[v] += ms / 40;
        }
        std::vector<float> S1(size_t(keys) * NC), S2(S1.size());
        cudaMemcpy(S1.data(), dS1, S1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(S2.data(), dS2, S2.size() * 4, cudaMemcpyDeviceToHost);
        double maxd = 0, e1 = 0, e2 = 0, r2 = 0; int n = 0;
        for (size_t i = 0; i < S1.size(); ++i) maxd = std::fmax(maxd, std::fabs(S1[i] - S2[i]));
        for (size_t i = 0; i < S1.size(); i += 997) {
            int k = i % keys, c = i / keys; double ref = 0;
            for (int d = 0; d < D; ++d) ref += double(K[size_t(k) * D + d]) * Q[size_t(c) * D + d];
            ref *= sc; e1 += (S1[i] - ref) * (S1[i] - ref); e2 += (S2[i] - ref) * (S2[i] - ref); r2 += ref * ref; ++n;
        }
        const double fl = 2.0 * keys * NC * D;
        printf("keys=%d TN %.3f ms (%.2f TFLOPS) NN %.3f ms (%.2f TFLOPS) speedup %.3f | max|TN-NN| %.3g rmse_vs_fp64 TN %.3g NN %.3g (rms ref %.3g, n=%d)\n",
               keys, t[0], fl / t[0] / 1e9, t[1], fl / t[1] / 1e9, t[0] / t[1], maxd, std::sqrt(e1 / n), std::sqrt(e2 / n), std::sqrt(r2 / n), n);
        cudaFree(dK); cudaFree(dKT); cudaFree(dQ); cudaFree(dS1); cudaFree(dS2);
    }
    return 0;
}
