# Night 13→14 Nov 2026: M6000 35B plan (cut 08:00 CET 14 Nov)

Advisor-confirmed 13 Nov ~16:00. Objective unchanged: makespan of two agentic slots at ~90–98K
(tool-result appends), not tok/s. Gates: see handbook §3. Output dir: `N2=$M/occamy35-night-20261114`
(create it; copy `RULES.md` from `occamy35-night-20261113/` and update the prod line to the current state).

## Starting point (see STATE.md for numbers)
- Prod after 13 Nov swap: BUNDLE3 + FA Maxwell GEMM prefill, **F16 KV, ub1024, cap 0**. ub2048 and cap1024 were
  NOT adopted: with ub2048 GEMM's `q_gt_1024` guard rejects the batch (-12.35% vs -15.76%), cap1024+GEMM is untested.
- K-only Q8 KV was **not adopted**: the GEMM path rejects q8_0 K (`reasons=ktype`), so with `-ctk q8_0`
  GEMM is off and +4K@90K = 30.47 s vs 23.09 s.
- Source: `occamy35-night-20261113/WP0/src` branch `gemm-b3` (HEAD abd5584-based); harness `WP0/harness`
  (n3_replay.py `--full --single`, wp0_report.py, wp0_gpu.sh = detached template with trap).

## Cards, in GPU order

| # | Card | Model | GPU? | Accept | Time cap |
|---|---|---|---|---|---|
| 1 | **WP3b: GEMM accepts Q8 K.** In the GEMM pack step, dequantise the q8_0 K strip to FP32 (or FP16) instead of falling back. Keep the `ktype` reason for other types. | opus (code) → sonnet (run) | yes | with `-ctk q8_0 -ctv f16`: engaged>0 on both slots; single +4K@90K ≤ gemm-F16 +3% (≤23.8 s); operator RMSE vs FP64 ≤ stock at 32K/98K; slot save/restore round trip with -ctk q8_0 gives 16 identical tokens; peak VRAM printed | 3 h of GPU-side iteration, then park and go to card 2 |
| 2 | **WP1: multi-stream GEMM**, only if the 13 Nov two-slot run shows `ne3` fallbacks (see STATE.md). Loop the GEMM per `ne[3]` slice and reuse the scratch serially. | opus, coded **in parallel with 1** | yes | N3 replay 3 blocks vs new prod: prefill s down; engaged both slots; RMSE vs FP64 ≤ stock per stream | 3 h |
| 3 | **WP2: fused mask+softmax epilogue** (online softmax across key chunks; spec in WRAPUP_20261113.md WP2) | opus | yes | `masked_softmax` share <6%; +4K@90K ≤ 20.5 s; FP64 ratio not worse | rest of night |
| 4 | **Q4 KV short rows**: first a K+V-flat diagnosis (nvprof <1,1> vs <1,0>, V dequant path), then K8V4 and K4V4 vs F16 under the reformed gate | sonnet run; opus only if a kernel fix is needed | short rows | dmean vs F16 within reorder floor, task parity, VRAM saved | ≤90 min GPU |
| 1b | **G1: raise GEMM `q_gt_1024` guard** (tile the Q dimension in 1024-row chunks, or loosen the guard if scratch fits) so ub2048 engages | opus (small, code during card 1) | yes, short | ub2048 single +4K@90K < ub1024 gemm (23.09 s); two-slot makespan better than -15.76%; peak VRAM ≤ 23.9 GiB with ≥600 MiB headroom; RMSE vs FP64 ≤ stock | 60 min GPU |
| 1c | **C1: cap1024 + GEMM** (flag only, no code): N3 replay prod vs prod+`--mixed-batch-size 1024` | sonnet run | yes | makespan not worse (CI), max token gap down (cap1024 alone: -48.6%) | 45 min GPU, slot it after card 1 |
| 5 | **WP4: moe_v2 occupancy** (≤64 regs, ≤24 KB smem, grid 96; bit-exact to BUNDLE3) | opus, CPU-side in parallel all night | only the final check | moe ms/step -20% two-slot deep, bit-exact logits, shallow t/s +4% | build by 06:00, GPU check if a slot is free |

WP5 (k=2 rows) stays off.

## Scout techniques per card (from KERNEL_INSPIRATION.md)
- WP3b / Q4 rows: magic-number int-to-float `(0x4B000000|b) - 2^23`, scale after the dot; dequant once into the pack smem.
- WP1: templated tiles + edge predication (maxDNN loses on non-64 shapes); serial scratch reuse per `ne3` slice.
- WP2: keep S in registers, P in smem; two-pass fallback if regs explode. ofa_decode later: Flash-Decoding split-K ≥2K keys + LSE reduce.
- WP4: `__launch_bounds__(128,4)`; bank = reg%4 matters only at SASS level, stay at ptxas level.

## Rules for the night
- One GPU job at a time under `gpu-bench.lock`. Coding cards run CPU-side while the GPU is busy.
- Any card that passes the reformed gate **and** the ≥1% release gate may replace prod with the same swap
  procedure: new container name, keep the previous one as `llamacpp-m6000-<tag>-prev`, never remove, and smoke-test
  one request through llama-swap, then check the `occamy_fa_gemm: engaged=` line in the log.
- Push each finished card's RESULT.md to gh-pub `evidence-20261112` under
  `data/moonshot-evidence/20261114T000000Z-m6000-35b-night/<card>/` after redaction and the privacy grep.
- Advisor: one table plus one question per milestone (card done, prod swap, blocker). Owner: morning report at 08:00.
- 07:30: stop launching GPU work; 08:00 push NIGHT_REPORT.md with a table, the prod state and the rollback commands.
