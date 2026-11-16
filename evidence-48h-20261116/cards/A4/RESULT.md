# A4 GPU results (run <W>/A4/run-20261116T060759Z, 15-16 Nov 2026)

| check | gate | result |
|---|---|---|
| test-backend-ops MUL_MAT_ID, flag on MIN=1 | engaged >0, 0 FAIL | PASS: 934/934 passed, 0 FAIL, engaged lines 6 |
| 16K prefill MoE ms/ubatch off vs on (nvprof, n=10) | >=1.07x | FAIL: 1387.2 vs 1353.7 ms = 1.0248x (busy 3775.3 vs 3755.6 = 1.0053x); on engaged 10 lines, off 0 |
| N3 short ABBA makespan total | no regression | A 158.43 / 158.49 (mean 158.46 s), B 159.08 / 158.56 (mean 158.82 s); B-A = +0.36 s (+0.23%), within the A-A spread plus 0.06 s, effectively neutral, no gain |
| CHECK=64 (first B arm, 64 lines) | nonfinite 0, relRMSE ~5e-3 | nonfinite 0 PASS; max relRMSE 2.323e-02, max maxabs 1.283e-01 (about 4x above the 5e-3 expectation; first lines 1.1e-2..2.2e-2) |
| VOID B arms | none | none: B_2 engaged 11 lines, B_3 engaged 11 lines; A arms 0 as expected |

Verdict: correct and engaged, but in-model gain 2.5% < 1.07x gate; no end-to-end makespan gain. Not adoptable on this evidence. Model dmean/KLD run was not part of this script.

## Method
a4_gpu.sh (unchanged), env N3_GPU and N3_UNLOAD_URL passed inline only; POST unload of qwen36-35b-moe, lock gpu-bench.lock, containers na4-*, prod not restored. Steps timeline: <W>/A4/run-20261116T060759Z/steps.log. Prof: <W>/A4/run-20261116T060759Z/prof/moe_gate.txt. Check: <W>/A4/run-20261116T060759Z/check.log. ABBA: <W>/A4/run-20261116T060759Z/{A_1,B_2,B_3,A_4}/analyze.log.

## Caveats
- ABBA x1 only, so no variance estimate beyond the A-A spread (0.06 s).
- CHECK compares FP32 against mmq (q8_1 error), so relRMSE reflects mmq error; 28e synthetic showed 5e-3..7.6e-3 for mmq, in-model values are higher (2.3e-2 max).
- Earlier CPU-build table kept in RESULT.cpu-pre.md.bak.
# A4: MoE FP32 prefill path (opt-in, default off)

| check | gate | result |
|---|---|---|
| CPU build (sm_52, incremental) | rc=0, no spills | PASS: rc=0, 0 spills (regs 80/144/95/108) |
| default-off behaviour | path unreachable without flag | PASS by construction: the eligibility check returns false when `GGML_CUDA_MOE_FP32_PREFILL` is unset |
| test-backend-ops MUL_MAT_ID, flag on, MIN=1 | engaged >0, 0 FAIL | PENDING (a4_gpu.sh step 1) |
| in-model MoE ms/ubatch, nvprof, 16K prefill, off/on | >=1.07x | PENDING (step 2) |
| CHECK relRMSE FP32 vs mmq (first B arm) | nonfinite 0, ~5e-3 (= mmq q8_1 error) | PENDING (step 3) |
| N3 short ABBA makespan, B vs A | no regression; B engaged >0 (else VOID) | PENDING (step 3) |
| operator RMSE vs FP64 | <= base | 28e mb: 3.7e-7..7.8e-7 vs 5.3e-3..7.6e-3 for mmq (synthetic) |
| model dmean | within same-run reorder floor | NOT IN SCRIPT; needs a separate KLD run |

Branch `a4-moe-fp32`, commit 622f643 on bundle9 52c9e84. Hashes are in RECEIPT.txt.

## What
- The 28e kernel (BM128 dequant-in-smem FP32 SGEMM, BK16, double-buffered) moves into `mmid.cu` as `moe_fp32_q40`.
- It is launched as the 28e "tail-bk16-mb3" split:
  - BN64 for full tiles plus tails >32;
  - BN32 for tails 17..32;
  - BN16 for tails 1..16.
  - The split reuses `ggml_cuda_mmq_make_tail_tiles`.
- It is hooked before the mmq branch in `ggml_cuda_mul_mat_id`.
- Eligibility:
  - sm_52 only;
  - Q4_0 weights with rows %128 == 0 and K %32 == 0;
  - F32 src1 and dst, 16-byte aligned;
  - at most 256 experts;
  - tokens >= `GGML_CUDA_MOE_FP32_PREFILL_MIN` (default 512).
- Modes:
  - flag=1: tail-bk16-mb3 for all mats.
  - flag=2: the TN8 variant (128 thr) for non-broadcast (down) full tiles.
- Debug:
  - `GGML_CUDA_OCCAMY_DEBUG`: prints `moe-fp32: engaged launches N ...` at powers of two.
  - `GGML_CUDA_MOE_FP32_CHECK=N`: also runs mmq into scratch for the first N ops and prints relRMSE, maxabs and nonfinite. This syncs the stream, so it is only for checking.

## Risks
- 28e was only 1.09x on synthetic layer shapes, so the 1.07x in-model gate has thin margin.
- q8_1 quantize savings (~0.2 ms/op) are already counted in the mmq-side total.
- Output is not bit-equal to mmq. It is about 10^4x closer to FP64, so model dmean shifts. The reorder-floor KLD run decides.
- Grids are sized to the maximum tile count, which leaves empty CTAs:
  - t64: up to T*8/64 + E;
  - t32 and t16: E each.
  - Each empty CTA exits early, costing a few us per op.
- Small-T (512..1024, mixed batch) efficiency is unmeasured. MIN can be raised to 2048 if the N3 two-slot rows lose.
