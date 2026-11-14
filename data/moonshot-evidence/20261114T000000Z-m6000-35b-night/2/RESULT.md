# Card 2 multi-stream GEMM — RESULT (correctness only; no N3 timing yet)

| check | result |
|---|---|
| test-backend-ops tbo_ms (ne3=2, q 64..1024, kv 4096..12544) on M6000 | all OK |
| ne3=2 output vs per-slice ne3=1, bit-for-bit (synthetic) | bit_identical_all=1 |
| RMSE vs FP64 (synthetic) | ~1.3e-08 (ms) vs ~1.3e-08 (stock) |

- Branch ms ff82bfd, GGML_CUDA_FA_MAXWELL_GEMM_MS=1. Residual ne3 loss with S1 is already 0%, so card 2's gain case is mostly covered; N3 timing deferred unless the stack misses 156 s. Run: <LAB>/occamy35-night-20261114/2/gpu-*
