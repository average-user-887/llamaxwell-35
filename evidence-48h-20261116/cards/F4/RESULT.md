# F4 single-slot row, prod batch8 libs (PENDING: script launched, waiting on gpu-bench.lock)

| row | libs | prompt step | prompt_n | prompt_ms | tok/s | result |
|---|---|---|---|---|---|---|
| single_1..3 prod:0 | prod-libs/batch8-20261115 | 84K restore -> 90K, -> 94K | pending | pending | pending | pending |

## Method
- `f4_gpu.sh` (copy of R0/gpu_r0.sh lock/unload/cleanup logic): flock -w 43200 on gpu-bench.lock, POST unload of qwen36-35b-moe only (N3_GPU / N3_UNLOAD_URL from launch env, not in files), then n3_replay `--single --single-only` with N3_SINGLES=prod:0, x3 sequential repeats, profile short (84K slot).
- Arm prod = batch8 env from run-35b-batch8-20261115.sh + GEMM, cap1024, ub2048, b2048, PREFILL_SINGLE_STREAM, OCCAMY_DEBUG. GGML_CUDA_MAXWELL_GRAPHS unset (and asserted absent). 250 W untouched. No MTP arm.
- Engagement counters (n2gdn variant/launches, MOE_V3 launches, FA gemm check) grepped into run-*/steps.log; per-run json in run-*/single_N/runs/single_prod_s0/run.json.

## Caveats
- Harness gives prefill timing at 90K and 94K contexts (not exactly 96K); no decode tok/s row (requests use n_predict=1). Server is -np 2 -c 196608 with only slot 0 active.
- `n3_replay_nf4.py` = n3_replay.py with only the container prefix changed n28h- -> nf4-. Slots read from N3/slots (read-only).

## Files
run dir via F4/LATEST, F4/launch.log, F4/f4_gpu.sh, F4/n3_replay_nf4.py
