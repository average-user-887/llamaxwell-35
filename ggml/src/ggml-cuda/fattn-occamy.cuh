#pragma once
// Track F (native-occamy): GQA split-K FA decode kernel for sm_52, env-gated (see fattn-occamy.cu).
#include "common.cuh"

int  ggml_cuda_occamy_fa_mode();
bool ggml_cuda_occamy_fa_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_occamy_fa_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool kvq8 = false);
// Q8N: native q8_0 K and/or V (GGML_CUDA_OCCAMY_KVQ8=1, needs PIPE=3, ADDR=1)
bool ggml_cuda_occamy_kvq8_enabled();
bool ggml_cuda_occamy_kvq8_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa_kvq8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_occamy_fa_2call_enabled();
bool ggml_cuda_occamy_fa_2call_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa_2call(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// MRFA: native multi-row F16 split-K decode for Q->ne[1] in 2..4 (GGML_CUDA_OCCAMY_FA_MROW=1; =2 compare vs stock)
int  ggml_cuda_occamy_fa_mrow_mode();
bool ggml_cuda_occamy_fa_mrow_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa_mrow(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// FA2: MROW=3 -> n_q == 2 uses the head-pair dual-row split-K kernel (n_q 3..4 stay on the MROW=1 kernel)
bool ggml_cuda_occamy_fa_mrow_is_fa2(const ggml_tensor * dst);
void ggml_cuda_occamy_fa_mrow_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// Approximate prefix block selection; =2 compares with full attention and retains stock.
int ggml_cuda_occamy_fa_sparse_mode();
bool ggml_cuda_occamy_fa_sparse_applicable(const ggml_tensor * dst);
void ggml_cuda_occamy_fa_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_occamy_fa_sparse_compare(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
