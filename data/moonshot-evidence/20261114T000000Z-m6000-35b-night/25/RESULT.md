# Card 25 RESULT: noise floor + fused-off row (Q)

| arm | config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | batch8 (FUSED=1) | 3 | 159.52 | - | 5.61 / 5.61 | 23867.00 |
| A2 | identical to A (noise floor) | 3 | 159.78 | 0.17% | 5.61 / 5.61 | 23867.00 |
| Q | A + GGML_CUDA_FA_MAXWELL_GEMM_FUSED=0 | 2 | 168.33 | 5.52% | 6.06 / 6.07 | 23867.00 |

Noise (A vs A2 delta): 0.17%; A runs: 159.49, 159.51, 159.56; A2 runs: 159.94, 159.46, 159.95

**Q env: GGML_CUDA_FA_MAXWELL_GEMM_FUSED=0.** No q_tiled force env var exists in 9/bin source (q_tiled is only an engagement counter in fattn-maxwell-gemm.cuh; tiling auto-engages at ub2048).

## Per run

| run | config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|---|
| A_1 | batch8 (FUSED=1) | 159.49 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| A_6 | batch8 (FUSED=1) | 159.51 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| A_8 | batch8 (FUSED=1) | 159.56 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| A2_2 | identical to A (noise floor) | 159.94 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| A2_5 | identical to A (noise floor) | 159.46 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| A2_7 | identical to A (noise floor) | 159.95 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| Q_3 | A + GGML_CUDA_FA_MAXWELL_GEMM_FUSED=0 | 168.13 | 6.05 | 23867.00 | (170, 0, 65536) | yes |
| Q_4 | A + GGML_CUDA_FA_MAXWELL_GEMM_FUSED=0 | 168.53 | 6.07 | 23867.00 | (170, 0, 65536) | yes |

## Method
Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A A2 Q Q A2 A A2 A. 9/bin all arms, env GEMM_FUSED=1 (Q: 0) PREFILL_SINGLE_STREAM=1 MOE_V3=1 OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset, cap1024 ub2048 -b 2048. Valid = gemm and moe_v3 engaged; fused calls>0 for A/A2, =0 for Q. Lock flock gpu-bench.lock; prod unloaded via POST.

## Caveats
Q with FUSED=0 takes the non-fused GEMM attention path (quantifies fused kernel value), not a q_tiled force.

## Files
Run dir: <LAB>/occamy35-night-20261114/25/run-20261115T044941Z; harness: <LAB>/occamy35-night-20261114/25/gpu_ab.sh, h/, summ25.py
