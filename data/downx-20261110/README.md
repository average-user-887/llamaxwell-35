# GGML_CUDA_MOE_DOWNX verification packet (10 Nov 2026)

Text only, assembled from existing raw files. No GPU run, model load or service action was done for this packet (docker inspect/logs read-only, cuobjdump on CPU, sha256sum). Anything not run is marked NOT RUN.

## Status

| Item | Status |
|---|---|
| Identity / hashes / config | DONE (below, hashes.txt) |
| ABBA speed, mode 1 | PASS as measured: +1.08% mean, +1.17% median (n=10/arm), below the 3% gate |
| ABBA speed, mode 3 | measured: +0.98% mean, +1.01% median. Mode attribution comes from the 897a09c message and the script's DXON knob, not recorded in raw files |
| ABBA speed, mode 2 alone | NOT RUN (no evidence) |
| 64-token replay, quantized KL-base file | PASS: dx0 == dx1 byte-identical, both builds (i1, i2) |
| Per-step FP32 full-vocab logit hashes | NOT RUN |
| Compared-value count, NaN/Inf count | NOT RUN (only file size 31289356 B, identical) |
| Dispatch proof (counter/trace) | NOT RUN (indirect evidence only) |
| Compiled resources (cuobjdump) | DONE |
| Long context, graph replay (bench-side), state restore, two-slot | see section 6 |

## 1. Identity and config
- Baseline 81e5211; candidates ccc8539 (iteration 1, DX is a bool, only mode 1) and 897a09c (DX bits 1|2, modes 1/2/3). main at packet time 124599c. Both commits only touch ggml/src/ggml-cuda/occamy-fused.cu (+48/-4 for 897a09c).
- Dirty patch: the i2 library was built 18:09 from the worktree .claude/worktrees/downx before 897a09c was committed (18:12). Worktree is clean now (status empty). The patch hash of the uncommitted state at build time was not recorded: it is believed, not proven, equal to 897a09c.
- The i1 lib was built 18:02, before ccc8539 was committed (18:06): same caveat. The production lib is this i1 lib (identical SHA, hashes.txt).
- SHA-256 of libs, bench executable, GGUF, text input, KL-base files: hashes.txt.
- GPU: Quadro M6000 24GB, sm_52, 24 SMs. Driver: NVRM 580.178.04. Build toolchain (image local/m6000-build-tools:20261013): nvcc 12.8.93, gcc-14 14.2.0, cmake 3.28.3. Runtime image for bench: nvidia/cuda:12.8.1-devel-ubuntu24.04.
- Build: scripts/build.sh (cmake -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_BUILD_TYPE=Release, target ggml-cuda test-backend-ops). Logs: resources/build-i1.log, build-i2.log (the logs show only "Built target" lines; ptxas verbose output was not kept).
- Runtime flags (bench): env-bench.txt. Prod: scripts/run-35b-downx-20261110.sh (env + `-ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 -fa on` etc.). No credentials in either.
- A/B design: ONE executable (prof35/app/llama) and ONE library per mode, flag GGML_CUDA_MOE_DOWNX toggled per container (0 vs on). Not different binaries. Mode 1 used lib i1, mode 3 used lib i2 (different libs between modes).
- Running service: docker inspect llamacpp-m6000 has GGML_CUDA_MOE_DOWNX=1 and the DOWNX lib bind-mounted (prod-container-env.txt, prod-container-config.txt). Env set does not by itself prove the kernel is dispatched (section 4).

## 2. Raw speed
- All 20 values per mode: raw/mode1_i1/abba.txt, raw/mode3_i2/abba.txt (order per cycle A B B A = dx0 dx1 dx1 dx0, cycles 1..5). Script: scripts/gpu.sh (i1 variant: scripts/gpu.sh.i1), analysis: scripts/analyze-abba.sh, output analysis-abba-output.txt.
- Mode 1 (tg128 tok/s): dx0 mean 79.2904 median 79.2552 (79.184-79.554); dx1 mean 80.1502 median 80.1814 (79.933-80.443). The commit message numbers (79.255 -> 80.181) are the medians. Per-cycle delta +1.05..+1.11%.
- Mode 3: dx0 mean 79.2417 median 79.2286; dx1 mean 80.0211 median 80.0286; per-cycle +0.92..+1.03%.
- Each run = fresh container, model load, `bench -p 0 -n 128 -r 3` (3 repetitions inside the run; any built-in warmup is llama-bench default, not independently verified). No separate warmup run. Failures: none, 20/20 values parsed per mode.
- Caveat: b-<cycle>-<dx>.txt csvs are overwritten by the second run of each ABBA pair, so full csv (avg_ns, stddev) exist only for runs 3 and 4 of each cycle; abba.txt has every value. Example avg_ns: 1614267406 (79.29 tok/s). Run-level stddev_ts 0.005-0.19.
- Clock, temperature, power: NOT LOGGED (not available). No lock/flock evidence in raw files beyond the script comment.
- Both arms use ub 512 and the bench-only OCCAMY_DEBUG=1; prod uses ub 1024. Gain at prod settings is not measured here.

## 3. 64-token exact replay
- Command (scripts/gpu.sh): `perplexity -m occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -c 128 -b 1 -ub 1 --chunks 1 -f text.txt --seed 0 --kl-divergence-base logits-dx$dx.bin`, run for dx=0 then dx=1, fresh container each, rc=0 both. n_ctx 128, 1 chunk, so 64 scored positions (second half of the context). Logs: raw/*/g2-dx0.txt, g2-dx1.txt: PPL 15.8181 +/- 4.88640 in both.
- Compared: `cmp` of the two --kl-divergence-base files. These hold quantized log-probs (llama.cpp KL-base format), NOT raw FP32 logits. "G2 BITWISE EQUAL" printed for both builds; all four files share SHA-256 47d21d98... (31289356 B each, hashes.txt).
- Fixture: text.txt is public llama.cpp documentation text (207042 B, SHA in hashes.txt); not copied, only the first 128 tokens are used.
- NOT RUN: per-step raw FP32 full-vocab logit hashing, compared-value counts, NaN/Inf counts. Equal quantized outputs are strong but not a proof of bit-equal FP32 logits.
- The comparison was ordinary kernel default path (the replay used batch 1: not the prefill path).

## 4. Dispatch proof
NOT RUN: the code has no per-launch counter or trace for the DX kernels. The only DX log line is at init with OCCAMY_DEBUG=1: `occamy-fused: device 0: DOWNX variant 2 blocks/SM -> grid 48`, which reports the occupancy query result for the DX instantiation, not that it was launched. Indirect evidence only: a consistent +1% tg delta that vanishes when the flag is 0, the different register counts (88 vs 95) and different SASS (section 5). A Nsight/CUPTI kernel-name trace was not taken.

## 5. Compiled resources (cuobjdump --dump-resource-usage, sm_52; resources/)
| kernel moe_v2<PHASE,DX> | REG | static smem | stack/local |
|---|---|---|---|
| <0,0> base | 88 | 96 B | 0 |
| <0,1> | 95 | 96 B | 0 |
| <0,2> | 87 | 96 B | 0 |
| <0,3> | 95 | 96 B | 0 |
| <3,0> | 85 | 64 B | 0 |
| <3,1> | 89 | 64 B | 0 |
| <3,2> | 85 | 64 B | 0 |
| <3,3> | 89 | 64 B | 0 |
| <1,0> / <2,0> | 36 / 83 | 32 / 96 B | 0 |
(i2 lib; i1 lib has the bool equivalents <0,true>=95, <3,true>=89, matching; base 88 in the 13:49 reference lib too.)

Occupancy arithmetic (sm_52: 65536 regs/SM, register alloc unit 256 regs per warp, 96 KiB smem/SM, 256 B smem alloc unit, 64 warps/SM, 8 warps per block at 256 threads):
- Regs <0,0>: 88*32 = 2816 = 11 units -> 2816/warp, 22528/block -> floor(65536/22528) = 2 blocks.
- Regs <0,1>/<0,3>: 95*32 = 3040 -> 12 units = 3072/warp, 24576/block -> floor(65536/24576) = 2 blocks. (3 blocks would need <= 80 regs.)
- Smem: 41728 dynamic + 96 static = 41824 -> 41984 after rounding; 2 blocks = 83968 <= 98304; 3 blocks = 125952 > 98304 -> smem alone caps at 2.
- Result: 2 blocks/SM for base and all DX variants = 16 warps/SM = 25% theoretical occupancy; grid = 24 SM * 2 = 48 blocks. Matches the logged "24 SMs x 2 blocks/SM -> grid 48" for both base and DOWNX. So the DX variants cost registers but not occupancy. Occupancy values here are hand-computed, not from the CUDA occupancy calculator.
- SASS (sass/): instruction counts <0,0> 2606, <0,1> 2828, <0,2> 2600, <0,3> 2828; BAR.SYNC 10/9/10/9. Mode 1 adds ~140 SHFL (vs 55) and ~30 STS (63 vs 33): group sums from registers via butterfly, one barrier fewer. FFMA identical (359). excerpt-*-first150.txt are the first 150 SASS lines; diff-*.txt are the first 300 lines of the register-normalized diff vs base (down phase is not isolated by label; excerpts are not a down-phase-only slice). Raw cuobjdump files were not committed (size).

## 6. Coverage
| Area | Status |
|---|---|
| Long context | NOT RUN |
| Graph replay | Prod log shows "graphs reused" on short completions only (smoke); no graph-vs-eager comparison: NOT RUN |
| State restore (slot save/restore, checkpoints) | NOT RUN (checkpoints were created in prod, restore not tested) |
| Two-slot | SMOKE ONLY: prod container up with n_slots=2; slot 1 served one 13-token prompt / 8-token completion (76.75 tok/s), slot 0 served short completions at ~78 tok/s (smoke-prod-log-excerpt.txt). No concurrent two-slot correctness test |
| Prefill path (ub 1024 / ub1) | NOT RUN (replay used -b 1 -ub 1; bench -p 0) |
| Mode 2 alone | NOT RUN |
| Prod build with iteration 2 (modes 2/3) | NOT RUN: prod runs the i1 lib (mode 1 only); mode 3 measured on a bench lib |

## Files
README.md, hashes.txt, env-bench.txt, prod-container-config.txt, prod-container-env.txt, smoke-prod-log-excerpt.txt, analysis-abba-output.txt, scripts/ (build.sh, gpu.sh, gpu.sh.i1, analyze-abba.sh, run-35b-downx-20261110.sh), raw/mode1_i1, raw/mode3_i2 (abba, g2 logs, bench csv), resources/, sass/.
