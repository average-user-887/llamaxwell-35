# Card 26 RESULT: fine cap sweep (9/bin, GEMM+FUSED+SINGLE_STREAM+MOE_V3)

| arm | arm config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | cap1024 ub2048 b2048 (ref) | 2 | 159.89 | - | 5.62 / 5.62 | 23867.00 |
| B | cap896 ub2048 b2048 | 2 | 164.51 | 2.89% | 4.94 / 4.95 | 23867.00 |
| C | cap1152 ub2048 b2048 | 2 | 160.24 | 0.22% | 6.19 / 6.19 | 23867.00 |
| D | cap1280 ub2048 b2048 | 2 | 160.23 | 0.21% | 6.87 / 6.88 | 23867.00 |

Verdict: **no arm passes** (gate: >=0.5% faster than A, max gap <= 6.5 s, VRAM <= A+1 GiB)

## Per run

| run | arm config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|---|
| A_1 | cap1024 ub2048 b2048 (ref) | 159.79 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| A_8 | cap1024 ub2048 b2048 (ref) | 159.99 | 5.62 | 23867.00 | (170, 170, 65536) | yes |
| B_2 | cap896 ub2048 b2048 | 164.36 | 4.93 | 23867.00 | (230, 230, 65536) | yes |
| B_7 | cap896 ub2048 b2048 | 164.66 | 4.95 | 23867.00 | (230, 230, 65536) | yes |
| C_3 | cap1152 ub2048 b2048 | 160.29 | 6.19 | 23867.00 | (170, 170, 65536) | yes |
| C_6 | cap1152 ub2048 b2048 | 160.18 | 6.19 | 23867.00 | (170, 170, 65536) | yes |
| D_4 | cap1280 ub2048 b2048 | 160.05 | 6.85 | 23867.00 | (170, 170, 65536) | yes |
| D_5 | cap1280 ub2048 b2048 | 160.41 | 6.88 | 23867.00 | (170, 170, 65536) | yes |

## Per-run server.log batch lines (see steps.log)

- A_1: 2.30.938.691 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:52:05
- B_2: 2.01.446.121 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:55:15
- C_3: 2.26.100.616 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:58:48
- D_4: 1.56.920.632 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 07:01:51
- D_5: 2.15.243.887 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 07:05:12
- C_6: 2.10.147.950 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 07:08:29
- B_7: 2.14.355.720 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 07:11:52
- A_8: 2.27.621.077 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 07:15:26

## Method
Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A B C D D C B A. All arms 9/bin, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset. cap1024 ub2048. Arms (cap, ub, -b) per table. Runs lacking engagement are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
n_batch/n_ubatch per arm are in the steps.log n_ubatch lines. Makespan noise about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/26/run-20261115T052232Z; harness: <LAB>/occamy35-night-20261114/26/gpu_ab.sh, h/, summ26.py
