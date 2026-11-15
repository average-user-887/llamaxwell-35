# 28e: prefill MoE matmul (Q4_0 mmq) tuning on batch8. Microbench below gate; not integrated

| op (one ubatch: 2048 tok x 8 experts, 256 experts) | batch8 mmq (in-model trace) | best candidate (FP32-dequant SGEMM, 3-launch tails) | speedup | gate (>=1.25x) |
|---|---|---|---|---|
| gate or up (512x2048), uniform routing | 11.46 ms | 10.73 ms (8x4, BK16, minB3) | 1.07x | FAIL |
| gate or up, skewed routing (lognormal 0.8) | 11.46 ms | 10.83 ms | 1.06x | FAIL |
| down (2048x512), uniform | 11.66 ms | 10.20 ms (8x8, BK16) | 1.14x | FAIL |
| down, skewed | 11.66 ms | 10.48 ms (8x4, BK16, minB3) | 1.11x | FAIL |
| upper bound if A loads were free (probe, down) | 11.66 ms | 9.99 ms | 1.17x | FAIL |
| layer total gate+up+down | 34.58 ms | 31.66 ms | 1.09x | FAIL |

Accuracy (6000 sampled outputs vs FP64 reference, all variants):
candidate relRMSE 3.7e-7 to 7.8e-7 (reorder level); q8_1-activation path (the current mmq numerics) 5.3e-3 to 7.6e-3.
The FP32 path is about 10^4 times closer to FP64 than the current path, so accuracy is not the problem. Speed is.

## What was tried (each run held the GPU lock for under 20 s, <gpu>, 250 W unchanged, no graphs flag)
- v1 BM128 x BN64 tile, single launch: 12.3 ms (0.93x). 64-wide tiles reach only 69% fill.
- v2 J64/J32/J16 tail launches (same split as the prod tail map): 12.1 ms. BK16 (k-permuted half block): 10.6-10.8 ms.
- v3 warp laid out as 8x4 threads (fewer shared-memory wavefronts): within noise, so shared-memory bandwidth is not the limit.
- Probe with every CTA reading weight row 0 (loads hit cache): only 3-5% faster, so the global loads are not the limit either.
- v4 8x8 register tile (147 regs): down 10.20 ms, gate/up worse (11.35 ms).
- Limit: the CUDA C FFMA inner loop sustains about 3.3 TFLOPS out of about 6.8 peak. Reaching 9.2 ms needs about 3.75 TFLOPS on real work.
  That would take SASS-level register-bank and dual-issue scheduling (maxas-style), which is out of scope for this card.

## Notes
- Baseline figures come from the card-28 in-model trace. Candidate figures come from a synthetic isolated bench at the same shapes (real Q4_0 layout and routing statistics).
- Candidate times leave out the q8_1 quantize (about 0.2 ms per op) that the current path needs. Counting that saving moves the best down case to about 1.16x. Still below the gate.
- Nothing was integrated. No N3 row was run, and branch night15-mmq28e was not created.
- Next lever, if wanted: fuse gate+up (shared B tile, 1024 rows per CTA) plus the FP32 path. Estimated at most about 1.15x on gate/up. Or a hand-scheduled SASS SGEMM core.
