# Card 24 RESULT: stack7 (S) vs batch8 (M), N5

| arm | n valid | makespan mean s | delta vs S | max token gap mean / max s | VRAM peak mean MiB | engagement (gemm,fused,moe_v3) |
|---|---|---|---|---|---|---|
| S | 5 | 161.31 | - | 5.58 / 5.59 | 23459.00 | gemm+fused required |
| M | 5 | 159.93 | -0.86% | 5.62 / 5.64 | 23867.00 | gemm+fused+moe_v3 required |

Verdict: **not a gate candidate** (gate: delta <= -1%, max gap <= 6.5 s, VRAM <= +1 GiB)

## Per run

| run | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|
| S_1 | 161.25 | 5.59 | 23459.00 | (200, 200, 0) | yes |
| S_3 | 161.22 | 5.59 | 23459.00 | (200, 200, 0) | yes |
| S_5 | 161.43 | 5.58 | 23459.00 | (200, 200, 0) | yes |
| S_7 | 161.38 | 5.58 | 23459.00 | (200, 200, 0) | yes |
| S_9 | 161.28 | 5.57 | 23459.00 | (200, 200, 0) | yes |
| M_10 | 159.94 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| M_2 | 159.92 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| M_4 | 159.90 | 5.63 | 23867.00 | (170, 170, 65536) | yes |
| M_6 | 159.87 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| M_8 | 160.02 | 5.64 | 23867.00 | (170, 170, 65536) | yes |

## Method
5 pairs alternating (S,M x5), short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot. S: 7/bin, cap1024, ub1024, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1. M: 9/bin (d4c13eb), cap1024, ub2048, same env + GGML_CUDA_MOE_V3=1 + GGML_CUDA_OCCAMY_DEBUG=1 (needed for the moe_v3 launch log line; log only). MAXWELL_GRAPHS unset. Runs with absent engagement lines are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
DEBUG env on M only prints stderr lines. Short profile; makespan noise per card 9 about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/24/run-20261115T041532Z; harness: <LAB>/occamy35-night-20261114/24/gpu_ab.sh, h/

## Makespan spread (s)

| arm | mean | median | min | max |
|---|---|---|---|---|
| S | 161.31 | 161.28 | 161.22 | 161.43 |
| M | 159.93 | 159.92 | 159.87 | 160.02 |

Delta M vs S: mean -0.86%, median -0.85%. Max gap S 5.59 / M 5.64 s. VRAM S 23459 / M 23867 MiB (+408).
