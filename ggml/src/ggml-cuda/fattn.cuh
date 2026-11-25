#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// occamy_fa_gemm: print GEMM engaged/fallback counters (no-op unless GGML_CUDA_FA_MAXWELL_GEMM=1 was used).
void ggml_cuda_fa_maxwell_gemm_report_free();
