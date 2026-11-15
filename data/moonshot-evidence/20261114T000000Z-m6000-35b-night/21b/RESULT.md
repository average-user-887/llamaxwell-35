# Card 21b RESULT: cap/ub/-b variants (9/bin, GEMM+FUSED+SINGLE_STREAM+MOE_V3)

| arm | arm config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | cap1024 ub2048 b2048 (ref) | 2 | 159.72 | - | 5.61 / 5.61 | 23867.00 |
| B | cap1024 ub1536 b2048 | 2 | 161.97 | 1.41% | 5.57 / 5.58 | 23663.00 |
| C | cap1024 ub3072 b3072 | 2 | 159.65 | -0.04% | 5.58 / 5.59 | 24275.00 |
| D | cap768 ub2048 b2048 | 2 | 162.89 | 1.99% | 4.26 / 4.27 | 23867.00 |
| E | cap1536 ub2048 b2048 | 2 | 160.39 | 0.42% | 8.14 / 8.14 | 23867.00 |

Verdict: **no arm passes** (gate: >=0.5% faster than A, max gap <= 6.5 s, VRAM <= A+1 GiB)

## Per run

| run | arm config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|---|
| A_1 | cap1024 ub2048 b2048 (ref) | 159.54 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| A_10 | cap1024 ub2048 b2048 (ref) | 159.90 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| B_2 | cap1024 ub1536 b2048 | 162.10 | 5.58 | 23663.00 | (230, 230, 65536) | yes |
| B_9 | cap1024 ub1536 b2048 | 161.85 | 5.56 | 23663.00 | (230, 230, 65536) | yes |
| C_3 | cap1024 ub3072 b3072 | 159.58 | 5.59 | 24275.00 | (160, 160, 65536) | yes |
| C_8 | cap1024 ub3072 b3072 | 159.72 | 5.57 | 24275.00 | (160, 160, 65536) | yes |
| D_4 | cap768 ub2048 b2048 | 163.14 | 4.27 | 23867.00 | (230, 230, 65536) | yes |
| D_7 | cap768 ub2048 b2048 | 162.65 | 4.26 | 23867.00 | (230, 230, 65536) | yes |
| E_5 | cap1536 ub2048 b2048 | 160.59 | 8.14 | 23867.00 | (170, 170, 65536) | yes |
| E_6 | cap1536 ub2048 b2048 | 160.19 | 8.13 | 23867.00 | (170, 170, 65536) | yes |

## Per-run server.log batch lines (see steps.log)

- A_1: 2.04.848.012 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:51:38
- B_2: 2.29.452.512 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:55:14
- C_3: 2.00.344.388 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:58:25
- D_4: 2.20.906.175 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:01:54
- E_5: 2.41.666.797 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:05:43
- E_6: 1.55.479.635 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:08:45
- D_7: 2.07.480.211 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:12:00
- C_8: 1.58.418.283 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:15:09
- B_9: 2.05.972.995 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:18:20
- A_10: 2.29.178.343 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 06:21:56

## Method
Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A B C D E E D C B A. All arms 9/bin, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset. cap1024 ub2048. Arms (cap, ub, -b) per table. Failed arms (e.g. C OOM) shown as VOID. Runs lacking engagement are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
n_batch/n_ubatch per arm are in the steps.log n_ubatch lines. Makespan noise about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/21b/run-20261115T041538Z; harness: <LAB>/occamy35-night-20261114/21b/gpu_ab.sh, h/, summ21b.py
