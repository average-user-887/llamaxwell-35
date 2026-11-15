# Card 30 RESULT: host flag A/B (9/bin, GEMM+FUSED+SINGLE_STREAM+MOE_V3)

| arm | arm config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | ref (cap1024 ub2048 b2048, default host flags) | 2 | 159.47 | - | 5.60 / 5.60 | 23867.00 |
| B | --threads 4 --threads-batch 4 --poll 100 | 2 | 159.73 | 0.17% | 5.60 / 5.61 | 23867.00 |
| C | --poll 0 | 2 | 159.22 | -0.16% | 5.59 / 5.60 | 23867.00 |
| D | --prio 2 --poll 100 | 2 | 159.24 | -0.14% | 5.59 / 5.60 | 23867.00 |

Verdict: **no arm passes** (gate: >=0.5% faster than A, max gap <= 6.5 s)

## Per run

| run | arm config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|---|
| A_1 | ref (cap1024 ub2048 b2048, default host flags) | 159.81 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| A_8 | ref (cap1024 ub2048 b2048, default host flags) | 159.12 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| B_2 | --threads 4 --threads-batch 4 --poll 100 | 160.28 | 5.61 | 23867.00 | (170, 170, 65536) | yes |
| B_7 | --threads 4 --threads-batch 4 --poll 100 | 159.18 | 5.58 | 23867.00 | (170, 170, 65536) | yes |
| C_3 | --poll 0 | 159.27 | 5.59 | 23867.00 | (170, 170, 65536) | yes |
| C_6 | --poll 0 | 159.16 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| D_4 | --prio 2 --poll 100 | 159.28 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| D_5 | --prio 2 --poll 100 | 159.19 | 5.58 | 23867.00 | (170, 170, 65536) | yes |

## Per-run server.log batch lines (see steps.log)

- A_1: 2.29.588.831 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:19:29
- B_2: 2.30.592.546 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:23:06
- C_3: 1.59.496.904 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:26:12
- D_4: 2.08.149.136 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:29:26
- D_5: 1.58.237.720 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:32:30
- C_6: 2.02.798.661 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:35:39
- B_7: 1.54.077.406 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:38:39
- A_8: 2.00.979.587 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 18:41:45

## Method
Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A B C D D C B A. All arms 9/bin, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset. cap1024 ub2048. Host flags per table via N3_SERVER_ARGS hook (copy of harness h/n3_replay.py; appended to llama-server args). Base cap1024 ub2048 b2048. Runs lacking engagement are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
n_batch/n_ubatch per arm are in the steps.log n_ubatch lines. Makespan noise about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/30/run-20261115T171551Z; harness: <LAB>/occamy35-night-20261114/30/gpu_ab.sh, h/, summ30.py
