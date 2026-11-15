# Card 23 RESULT: flag variants (9/bin, GEMM+FUSED+SINGLE_STREAM+MOE_V3)

| arm | arm config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | ref | 2 | 159.37 | - | 5.58 / 5.60 | 23867.00 |
| B | no single-stream | 2 | 159.46 | 0.06% | 5.61 / 5.61 | 23867.00 |
| C | -b 4096 | 2 | 159.57 | 0.13% | 5.61 / 5.61 | 23867.00 |
| D | ckpt 8 / step 4096 | 2 | 159.78 | 0.26% | 5.60 / 5.60 | 23867.00 |

Verdict: **no arm passes** (gate: >=0.5% faster than A, max gap <= 6.5 s, VRAM <= A+1 GiB)

## Per run

| run | arm config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |
|---|---|---|---|---|---|---|
| A_1 | ref | 158.88 | 5.57 | 23867.00 | (170, 170, 65536) | yes |
| A_8 | ref | 159.85 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| B_2 | no single-stream | 159.65 | 5.61 | 23867.00 | (160, 160, 65536) | yes |
| B_7 | no single-stream | 159.27 | 5.60 | 23867.00 | (160, 160, 65536) | yes |
| C_3 | -b 4096 | 159.52 | 5.61 | 23867.00 | (160, 160, 65536) | yes |
| C_6 | -b 4096 | 159.62 | 5.61 | 23867.00 | (160, 160, 65536) | yes |
| D_4 | ckpt 8 / step 4096 | 159.60 | 5.60 | 23867.00 | (170, 170, 65536) | yes |
| D_5 | ckpt 8 / step 4096 | 159.96 | 5.60 | 23867.00 | (170, 170, 65536) | yes |

## Single-stream lines per run

- A_1: 2.00.157.847 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 04:50:12
- B_2:  04:53:45
- C_3: 2.37.154.145 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 04:57:29
- D_4: 2.11.832.198 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:00:47
- D_5: 2.41.911.666 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:04:35
- C_6: 2.17.358.178 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:07:59
- B_7:  05:11:16
- A_8: 2.19.489.799 I srv    operator(): prefill-single-stream: deferred slot prompt batches = 1| 05:14:42

## Method
Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A B C D D C B A. All arms 9/bin, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset. cap1024 ub2048. A=-b 2048 ref; B=no LLAMA_PREFILL_SINGLE_STREAM env (harness passes it as env only, no CLI flag); C=-b 4096 (N3_BATCH param); D=--ctx-checkpoints 8 --checkpoint-min-step 4096. Runs lacking engagement are void. Lock: flock gpu-bench.lock; prod unloaded via POST.

## Caveats
Harness gained N3_BATCH/N3_CKPT/N3_CKPT_STEP env params (defaults unchanged). Makespan noise about 0.5%.

## Files
Run dir: <LAB>/occamy35-night-20261114/23/run-20261115T034706Z; harness: /gpu_ab.sh, h/, summ23.py
