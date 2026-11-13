// N4 driver: feed dumped FA tensors (n4_dump output) to ggml_cuda_flash_attn_ext of the linked libggml-cuda.
// One process = one kernel arm: env GGML_CUDA_FA_MAXWELL_GEMM=1 selects GEMM, unset selects stock (flag is cached on first call).
// usage: n4_drive <dumpdir/stopS> <layer> <outdir> [stages <rows csv>]
//   writes <outdir>/L<layer>_O.f32 [nq,16,256] f32. With `stages` (needs GEMM semantics; runs the v4 stage pipeline from the
//   included header, with a softmax copy that also emits m,l) writes S.f32 (post-SGEMM, scale applied), P.f32, ML.f32 for the rows
//   and asserts the replicated pipeline output is bit-equal to the library call.
// Build: see N4_DESIGN.md section 7 (same line as integration-v4/build.sh probe step).
#include "ggml/src/ggml-cuda/fattn-tile.cuh"
#include "ggml/src/ggml-cuda/fattn.cuh"
#include "ggml/src/ggml-cuda/fattn-maxwell-gemm.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(3); } } while (0)
static std::vector<char> slurp(const std::string & p) { std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(4); } std::stringstream s; s << f.rdbuf(); std::string t = s.str(); return std::vector<char>(t.begin(), t.end()); }
static double jnum(const std::string & j, const char * key) { size_t p = j.find(std::string("\"") + key + "\":"); if (p == std::string::npos) { fprintf(stderr, "meta %s\n", key); exit(5); } return atof(j.c_str() + p + strlen(key) + 3); }
static ggml_tensor lay(ggml_type ty, int64_t a, int64_t b, int64_t c, size_t s0, size_t s1, size_t s2, void * d) {
    ggml_tensor t{}; t.type = ty; t.ne[0] = a; t.ne[1] = b; t.ne[2] = c; t.ne[3] = 1; t.nb[0] = s0; t.nb[1] = s1; t.nb[2] = s2; t.nb[3] = s2 * c; t.data = d; return t; }

// copy of maxwell_gemm_masked_softmax with extra (m,l) outputs; arithmetic identical to the header kernel (keep in sync, asserted by bit-equality)
static __global__ void n4_softmax_ml(float * scores, const char * mask, size_t mask_row_stride, int key_count, int query_count, int query_start, float * ml /*[64*8][2]*/) {
    const int lq = blockIdx.x, lh = blockIdx.y, query = query_start + lq;
    float * column = scores + (size_t(lh) * maxwell_gemm_tile_queries + lq) * key_count;
    if (query >= query_count) { for (int k = threadIdx.x; k < key_count; k += blockDim.x) column[k] = 0.0f; return; }
    const half * mr = mask ? reinterpret_cast<const half *>(mask + query * mask_row_stride) : nullptr;
    float lmax = -FLT_MAX, lsum = 0.0f;
    for (int k = threadIdx.x; k < key_count; k += blockDim.x) {
        const float v = column[k] + (mr ? __half2float(mr[k]) : 0.0f);
        if (v > lmax) { lsum = lsum * __expf(lmax - v) + 1.0f; lmax = v; } else { lsum += __expf(v - lmax); } }
    const float mx = maxwell_gemm_block_reduce<true>(lmax); lsum *= __expf(lmax - mx);
    const float tot = maxwell_gemm_block_reduce<false>(lsum);
    if (threadIdx.x == 0) { ml[(lh * 64 + lq) * 2] = mx; ml[(lh * 64 + lq) * 2 + 1] = tot; }
    if (tot == 0.0f) { for (int k = threadIdx.x; k < key_count; k += blockDim.x) column[k] = 0.0f; return; }
    const float inv = 1.0f / tot;
    for (int k = threadIdx.x; k < key_count; k += blockDim.x) column[k] = __expf(column[k] + (mr ? __half2float(mr[k]) : 0.0f) - mx) * inv;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
    const std::string dir = argv[1]; const int L = atoi(argv[2]); const std::string out = argv[3];
    const bool stages = argc > 4 && !strcmp(argv[4], "stages");
    std::vector<int> rows; if (stages && argc > 5) for (char * c = strtok(argv[5], ","); c; c = strtok(nullptr, ",")) rows.push_back(atoi(c));
    const auto mj = slurp(dir + "/meta.json"); const std::string meta(mj.begin(), mj.end());
    const int nq = (int) jnum(meta, "nq"), nqp = (int) jnum(meta, "nq_pad"), nk = (int) jnum(meta, "nk"); const float scale = (float) jnum(meta, "scale");
    const std::string pre = dir + "/L" + std::to_string(L) + "_";
    auto hq = slurp(pre + "Q.f32"), hk = slurp(pre + "K.f16"), hv = slurp(pre + "V.f16"), hm = slurp(dir + "/mask.f16");
    void *dq, *dk, *dv, *dm, *dout; CK(cudaMalloc(&dq, hq.size())); CK(cudaMalloc(&dk, hk.size())); CK(cudaMalloc(&dv, hv.size())); CK(cudaMalloc(&dm, hm.size()));
    const size_t out_bytes = (size_t) nq * 16 * 256 * 4; CK(cudaMalloc(&dout, out_bytes + 4096)); CK(cudaMemset(dout, 0xFF, out_bytes + 4096)); // NaN fill: unwritten rows visible
    CK(cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), hv.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm, hm.data(), hm.size(), cudaMemcpyHostToDevice));
    // layouts identical to the graph's FA node: Q f32 {256,nq,16} nb1=16*256*4 (token stride), nb2=256*4; K/V f16 {256,nk,2} compact; mask f16 {nk,nq_pad}
    ggml_tensor q = lay(GGML_TYPE_F32, 256, nq, 16, 4, 16 * 256 * 4, 256 * 4, dq);
    ggml_tensor k = lay(GGML_TYPE_F16, 256, nk, 2, 2, 512, (size_t) nk * 512, dk), v = lay(GGML_TYPE_F16, 256, nk, 2, 2, 512, (size_t) nk * 512, dv);
    ggml_tensor m = lay(GGML_TYPE_F16, nk, nqp, 1, 2, (size_t) nk * 2, (size_t) nk * 2 * nqp, dm);
    ggml_tensor o = lay(GGML_TYPE_F32, 256, 16, nq, 4, 256 * 4, 16 * 256 * 4, dout);
    o.op = GGML_OP_FLASH_ATTN_EXT; o.src[0] = &q; o.src[1] = &k; o.src[2] = &v; o.src[3] = &m;
    float opp[4] = {scale, 0.0f, 0.0f, 0.0f}; memcpy(o.op_params, opp, sizeof(opp));          // op_params[3]=prec F32 set by graph; [4]=n_kv_max
    { int32_t pr = GGML_PREC_F32; memcpy((char *) o.op_params + 12, &pr, 4); }
    ggml_backend_cuda_context ctx(0);
    CK(cudaDeviceSynchronize()); ggml_cuda_flash_attn_ext(ctx, &o); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<char> ho(out_bytes); CK(cudaMemcpy(ho.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
    { FILE * f = fopen((out + "/L" + std::to_string(L) + "_O.f32").c_str(), "wb"); fwrite(ho.data(), 1, out_bytes, f); fclose(f); }
    printf("arm=%s nq=%d nk=%d tail=%d wrote O\n", getenv("GGML_CUDA_FA_MAXWELL_GEMM") ? "gemm" : "stock", nq, nk, nk % 4096);
    if (!stages) return 0;
    // ---- replicated v4 stage pipeline for selected rows (GEMM only) ----
    const int D = 256, cols = 8 * 64, key_cap = 98304, split_n = (nk + 4095) / 4096, full = nk / 4096, tail = nk % 4096;
    float *kf, *vf, *pq, *sc, *part, *mlbuf, *dout2; CK(cudaMalloc(&kf, (size_t) key_cap * D * 4)); CK(cudaMalloc(&vf, (size_t) key_cap * D * 4));
    CK(cudaMalloc(&pq, (size_t) 1024 * 16 * D * 4)); CK(cudaMalloc(&sc, (size_t) key_cap * cols * 4)); CK(cudaMalloc(&part, (size_t) 24 * cols * D * 4));
    CK(cudaMalloc(&mlbuf, 512 * 2 * 4)); CK(cudaMalloc(&dout2, out_bytes)); CK(cudaMemset(dout2, 0xFF, out_bytes));
    const float **pl, **pr; float ** po; CK(cudaMalloc(&pl, 24 * 8)); CK(cudaMalloc(&pr, 24 * 8)); CK(cudaMalloc(&po, 24 * 8));
    cudaStream_t st = ctx.stream(); cublasHandle_t h = ctx.cublas_handle(); const float one = 1, zero = 0; const int padq = (nq + 63) / 64 * 64;
    maxwell_gemm_pack_queries<<<4096, 256, 0, st>>>((const char *) dq, pq, nq, padq, q.nb[1], q.nb[2]);
    maxwell_gemm_pv_pointers<<<1, 256, 0, st>>>(vf, sc, part, pl, pr, po, split_n);
    std::vector<float> S((size_t) rows.size() * 16 * nk), P((size_t) rows.size() * 16 * nk), ML((size_t) rows.size() * 16 * 2), col(nk), mlh(512 * 2);
    for (int g = 0; g < 2; ++g) {
        maxwell_gemm_convert_kv<<<65535, 256, 0, st>>>((const char *) dk + g * k.nb[2], (const char *) dv + g * v.nb[2], kf, vf, nk, k.nb[1], v.nb[1]);
        for (int qs = 0; qs < padq; qs += 64) {
            const int chunk = (qs / 64) * 2 + g;
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, nk, cols, D, &scale, kf, D, pq + (size_t) chunk * cols * D, D, &zero, sc, nk));
            std::vector<int> sel; for (size_t r = 0; r < rows.size(); ++r) if (rows[r] >= qs && rows[r] < qs + 64) sel.push_back((int) r);
            for (int r : sel) for (int hl = 0; hl < 8; ++hl)   // S: post-SGEMM (scale folded into alpha), before mask
                CK(cudaMemcpy(&S[((size_t) r * 16 + g * 8 + hl) * nk], sc + ((size_t) hl * 64 + (rows[r] - qs)) * nk, (size_t) nk * 4, cudaMemcpyDeviceToHost));
            n4_softmax_ml<<<dim3(64, 8), 256, 0, st>>>(sc, (const char *) dm, m.nb[1], nk, nq, qs, mlbuf);
            CK(cudaMemcpy(mlh.data(), mlbuf, 512 * 2 * 4, cudaMemcpyDeviceToHost));
            for (int r : sel) for (int hl = 0; hl < 8; ++hl) {
                CK(cudaMemcpy(&P[((size_t) r * 16 + g * 8 + hl) * nk], sc + ((size_t) hl * 64 + (rows[r] - qs)) * nk, (size_t) nk * 4, cudaMemcpyDeviceToHost));
                memcpy(&ML[((size_t) r * 16 + g * 8 + hl) * 2], &mlh[(hl * 64 + rows[r] - qs) * 2], 8); }
            if (full > 0) CUBLAS_CHECK(cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, cols, 4096, &one, pl, D, pr, nk, &zero, po, D, full));
            if (tail > 0) CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, D, cols, tail, &one, vf + (size_t) full * 4096 * D, D, sc + full * 4096, nk, &zero, part + (size_t) full * cols * D, D));
            maxwell_gemm_reduce_products<<<(cols * D + 255) / 256, 256, 0, st>>>(part, dout2, nq, qs, g * 8, split_n);
        }
    }
    CK(cudaDeviceSynchronize());
    std::vector<char> h2(out_bytes); CK(cudaMemcpy(h2.data(), dout2, out_bytes, cudaMemcpyDeviceToHost));
    const bool same = memcmp(h2.data(), ho.data(), out_bytes) == 0; printf("replica_bit_equal_to_library=%d\n", (int) same);
    auto w = [&](const char * n, const void * d, size_t b) { FILE * f = fopen((out + "/L" + std::to_string(L) + "_" + n).c_str(), "wb"); fwrite(d, 1, b, f); fclose(f); };
    w("S.f32", S.data(), S.size() * 4); w("P.f32", P.data(), P.size() * 4); w("ML.f32", ML.data(), ML.size() * 4);
    return same ? 0 : 6;
}
