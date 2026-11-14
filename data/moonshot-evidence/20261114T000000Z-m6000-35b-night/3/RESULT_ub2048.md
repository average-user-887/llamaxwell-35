# Card 3c: stack cap1024, ub1024 vs ub2048 (short profile, 2 slots, 3 alternating pairs)

| metric | ub1024 (base) mean [3 runs] | ub2048 (test) mean [3 runs] | delta |
|---|---|---|---|
| makespan total s | 161.16 (161.05/161.17/161.26) | 160.22 (159.93/160.43/160.30) | -0.94 s (-0.58%), every pair favours ub2048 |
| slot0 (agent0) completion s | 155.26 | 154.33 | -0.93 |
| slot1 (agent1) completion s | 136.16 | 135.21 | -0.95 |
| TTFT mean s | 17.82 | 16.91 | -0.91 (-5.1%) |
| max token gap s | 5.58-5.62 | 5.57-5.58 | none |
| peak VRAM MiB | 23459 | 23867 | +408 |
| engaged= (exit) | engaged=200 q_tiled=0 fallback~19280 | engaged=170 q_tiled=30 fallback~19313 | q_tiled path used at ub2048 |
| allocation failures | 0 | 0 | |

Method: card7_gpu.sh harness copy (<LAB>/occamy35-night-20261114/3/ub2048/h, only change: container prefix n3c-), arm "prod" (cap 1024, GEMM=1) with env GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1 LLAMA_PREFILL_SINGLE_STREAM=1, bin /7/bin, FULL_UB=1024/2048, order ub2048,ub1024 x3, short profile, no sweep/ctxfull. flock on gpu-bench.lock, unload by POST, GPU idx via env, MAXWELL_GRAPHS unset.
Caveats: card 7 base was cap 0 (161.0 s) so not reusable; base rerun here with cap 1024 gave 161.16 s (same). Gain ~1 s is small but consistent (n=3, spread ~0.3 s). test-backend-ops step skipped (same bin as card 7). Max gap is same 5.6 s (earlier card 7 analyze showed 10.9 s on one run, differing cap). Per-turn detail in each analyze.log.
Files: <LAB>/occamy35-night-20261114/3/ub2048/run-20261114T124157Z (steps.log, ub*_N/analyze.log, runs/*/server.log), <LAB>/occamy35-night-20261114/3/ub2048/card3c_gpu.sh
