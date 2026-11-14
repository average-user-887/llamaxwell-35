# Card 7c: model-level KLD gate, stack vs prod (GEMM only)

Reference = prod arm P. Floor = same-run reorder arm Pp (P + FA_PARALLEL_BLOCKS_MULT=2), measured against P in the same run.
Rows per bucket: 128 (2 docs x 64), teacher-forced full-vocab FP32 logits.

| bucket | S-vs-P dmean | floor Pp-vs-P dmean | S top agree vs P | floor top | dmean <= floor | top >= -0.3 pt |
|---|---|---|---|---|---|---|
| 32768 | +0.00199 | +0.00531 | +0.00 pt | -1.56 pt | PASS | PASS |
| 65536 | +0.00034 | +0.00042 | +0.00 pt | +0.00 pt | PASS | PASS |
| 98304 | +0.00150 | +0.00128 | -0.78 pt (1 row of 128) | -0.78 pt (1 row) | FAIL (+.00022, inside noise) | FAIL (1 row; floor identical) |
| pooled 384 rows | +0.00128 (se .00029) | +0.00234 (se .00074) | -0.26 pt | -0.78 pt | PASS | PASS |

Operator RMSE vs FP64 not worse than stock: from card 7b (GATES_RESULT.md), fused ~= stock at kv8448 and ~0.5x stock at 32768/98304: PASS under the reformed gate (not re-run here).

Verdict: pooled PASS on both model lines. Strict per-bucket reading FAILs only at 98304, by +.00022 dmean (floor +.00128 is itself the same size; per-doc: doc0 S .00299 vs floor .00256, doc1 both ~0) and by one top-token flip in 128 rows, which the floor arm also shows. A pure reorder perturbation does the same, so 98304 is not separable from the floor (N2 finding, 13 Nov). Sample is only 2 docs, so doc-level CIs are not usable (analyzer printed nan).

## Method
Harness: G1c teacher-forced rows (g1c_rows from NC2/bin, G1c method), copy in kld/kld_harness.py (containers n7c-*), analyzer kld/kld_analyze.py wrapping g1c_analyze.pair. Libs dir = 7/bin for all arms; only libggml-cuda.so differs: P/Pp = 0/bin (GEMM), S = 7/bin (GEMM + FUSED). Env = BUNDLE3 env (identical to G1c BASE_ENV+FA_ENV) + GGML_CUDA_FA_MAXWELL_GEMM=1 (+GEMM_FUSED=1 for S). F16 KV, ub 1024, fa on. GGML_CUDA_MAXWELL_GRAPHS unset. One lock hold (flock gpu-bench.lock), POST unload qwen36-35b-moe, 39 min GPU. Engaged: all arms engaged=1860; S fused calls=1860 softmax_launches=59440.

## Caveats
- g1c_rows is single-sequence: server-only parts (--prefill-single-stream, cap 1024) cannot act and were not exercised; they change scheduling/batch composition, not kernel numerics. ub1024 applied.
- Reference is prod, not FP64 (no model-level FP64 exists); floor from one perturbation arm (Pp), docs 0-1 only.
- Operator RMSE line is cited from 7b, not re-measured.

## Files
<LAB>/occamy35-night-20261114/7/kld/run-20261114T144727Z/ (KLD_TABLE.md, analyze.out, steps.log, P/ S/ Pp/ f32 rows + logs), <LAB>/occamy35-night-20261114/7/kld/{run_kld.sh,kld_harness.py,kld_analyze.py}
