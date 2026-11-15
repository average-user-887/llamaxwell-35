# Card 16 RESULT: N3 stack (S) vs stack-moe3+ub2048 (M)

| arm | n valid | makespan mean s | delta vs S | max token gap mean / max s | VRAM peak mean MiB | engagement (gemm,fused,moe_v3) |
|---|---|---|---|---|---|---|
| S | 5 | 161.55 | - | 5.59 / 5.60 | 23459.00 | gemm+fused required |
| M | 5 | 159.63 | -1.19% | 5.60 / 5.61 | 23867.00 | gemm+fused+moe_v3 required |

Verdict: **GATE-CANDIDATE** (gate: delta <= -1%, max gap <= 6.5 s, VRAM <= +1 GiB)

## Per run

| run | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|
| S_1 | 161.50 | 5.60 | 23459.00 | (200, 200, 0) | yes |
| S_4 | 161.70 | 5.59 | 23459.00 | (200, 200, 0) | yes |
| S_5 | 161.69 | 5.59 | 23459.00 | (200, 200, 0) | yes |
| S_8 | 161.46 | 5.58 | 23459.00 | (200, 200, 0) | yes |
| S_9 | 161.40 | 5.59 | 23459.00 | (200, 200, 0) | yes |
| M_10 | 159.59 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| M_2 | 159.46 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| M_3 | 159.99 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| M_6 | 159.63 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| M_7 | 159.50 | 5.61 | 23867.00 | (170, 170, 65536) | yes |

## Method
5 ABBA pairs (S,M,M,S,S,M,M,S,S,M), short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot. S: 7/bin, cap1024, ub1024, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1. M: 9/bin (d4c13eb), cap1024, ub2048, same env + GGML_CUDA_MOE_V3=1 + GGML_CUDA_OCCAMY_DEBUG=1 (needed for the moe_v3 launch log line; log only). MAXWELL_GRAPHS unset. Runs with absent engagement lines are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
DEBUG env on M only prints stderr lines. Short profile; makespan noise per card 9 about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/16/run-20261114T232838Z; harness: <LAB>/occamy35-night-20261114/16/gpu_ab.sh, h/
