# M6000 35B night report, 12->13 Nov 2026

Lead: Claude Opus session, directed by the Fable advisor. The night queue came from the owner's auditor.
Rule: measurement only. No adoptions, no prod changes, no pushes to main. Evidence goes on branch `evidence-20261112` only.
Objective: end-to-end makespan of two agentic workflows near 98,304 tokens. Prod has GEMM off and cap 0. Cut was 08:00 CET.
The objective rests on the intended workload, not the observed one: per NC1, prod has seen almost no deep-append traffic yet (0.13% of time).

## Prod state
Unchanged all night: BUNDLE3, `-np 2`, F16 KV, GEMM off, graphs off. The rollback container is kept.

## Results

| id | what | result | status |
|---|---|---|---|
| NC1 | workload census, 150 requests 10-12 Nov | Mostly lab harness, about 0.3% busy. Time shares: decode 56.6%, shallow prefill 25.8%, full reprocess 12.4%, restore >=5.2%, deep prefill 0.13%. No prefill over 1K started past 32K depth. Cache reuse 93.8%. The logs can't set replay weights, so the replay design stays the weight source | done |
| NC2 | G1c harness audit | Latent bug: DOCLEN 98368, so NR=128 at 98K read 64 tokens past the end. The real run used nr=64 and is unaffected. Fixes: harness slice, plan validation, rc 16/17 on the rows binary, and an analyzer that exits 2 on partial or NaN data (15 tests) | done |
| NC3 | N4 design: GEMM per-op dumps, F64 reference | Design, reference self-test PASS, tools built. G1c scored rows are Q=1 decodes that run stock in both arms, so the G1c KL is propagated drift | design only, N4 moved to next session |
| N1 | deep-append profile, 90K restored slot, nvprof, 2 reps | GEMM off: +1K 7.97 s, +4K 31.98 s; flash_attn_tile<256,256,4,8> is 78% of prefill GPU time. GEMM on (v4 lib): +1K 5.84 s, +4K 23.39 s (about -27%); attention GEMMs about 69%. Restore 0.56 s, first token about 0.10 s; prefill is >98% of the append cost | done; arm B is the v4 lib, not a pure flag flip |
| Q8S | q8 KV quality vs F16 arm A (docs 0-3, 256 rows/bucket; shallow 272 rows) | K+V: +.00305/-1.17, +.00126/-0.39, +.00140/-0.78 (32K/65K/98K), shallow +.0052/-1.47. K only: +.00216/-0.39, +.00090/-0.39, +.00097/0.00, shallow +.0026/-0.74. V only: shallow +.0025/-0.74. KV memory at -np 2 -c 196608: F16 3840 MiB, K or V only 2940 MiB, K+V 2040 MiB (-1800) | strict gate FAIL in all arms (advisor ruling). Not adoptable |
| N2 | multi-doc noise floor Ap = A + FA_PARALLEL_BLOCKS_MULT=2, docs 0-7, plus shallow | Ap vs A: +.00331/-0.98, +.00137/-0.39, +.00263/-0.98; shallow +.00224/-1.10. A repeat on shallow: exactly 0 (bit-exact). GEMM B vs A: +.00345/-1.17, +.00253/-0.78, +.00332/-0.98 | see readings below |
| Q8N | native q8 K/V read in ofa_decode_hs_ar (flag GGML_CUDA_OCCAMY_KVQ8=1, default off) | G0 PASS: 127/121/118 regs for K+V/K/V, 0 spill, 2 blocks/SM as F16; flag-off SASS 7367/7367 functions identical; CPU lane check 8.4M values, 0 mismatch. Patch: Q8N/q8n-abd5584.patch | GPU gates NOT_RUN (see below) |
| N3 | two-workflow replay, makespan | Harness and short profile ready (4 turns, 84K -> ~95.8K, Williams 5 blocks; arms BUNDLE3, +cap1024, GDN4) | NOT_RUN |

## Readings
- GEMM prefill fails the owner gate (dmean <= +.0010, same-top >= -.3) in every bucket. It cannot be separated from the 8-doc Ap floor: B minus Ap is +.0001 to +.0012 dmean and -0.2 to -0.4 pt, and the CIs overlap.
- Ap itself fails the gate at all depths, so no benign change in accumulation order can pass this gate at 32K-98K. The owner decides whether a floor-relative reading applies. The gate was not relaxed.
- q8 shallow drift is real, not run noise, because the A repeat is exactly 0. K-only and V-only drift about as much as the Ap reorder. K+V drifts about 2.3x as much as Ap.
- The deep-append prefill lever is attention: 78% stock tile, 69% of the GEMM arm. A native q8 or packed tile read is the phase-2 prefill candidate.

## Not run, and why
Q8N GPU gates, N3, N4 and N5 did not run. The Q8N GPU card stopped at launch when the cloud quota ran out (5-hour limit), before it took the GPU lock. The session resumed after the 08:00 cut. Q8S also left its add-on checks undone after it overran its GPU budget (90 of 75 min): rotation-off arm, two-slot independence, q8 slot round-trip, slot size and restore time, peak VRAM under fill.

Next session, in the advisor's order:
1. Q8N GPU (exactness <1,1> and <1,0> at 32K/98K on 4 docs, slot-restore test, two-slot deep ABBA x3 vs F16, nvprof).
2. N3 replay.
3. N4.
4. N5 (b12318 smoke).

## Where things are
- This dir has one subdir per card. Each has RESULT.md or REPORT_TABLES.md and the scripts that ran.
- Excluded here: binaries, logits/state dumps and source clones; they stay on lab storage.
- N1 nvprof traces are gzipped CSV.
- Redactions: `<LAB>` lab storage root, `<MODELS>` models root, `<HOME>` home, `<lab-endpoint>` and `<gpu>` replace endpoint and device ids.

## Day 13 Nov addendum (09:10-13:57 CET, GPU runs; subdir `day13/`)
Runs: the Q8N GPU script (Claude card, detached) plus N3, N4 and the stock baseline (run by another lab agent while the Claude quota was out). The lead reviewed every number below against the files. Prod is unchanged (BUNDLE3); lock free, no test containers left.

| id | result | status |
|---|---|---|
| Q8N exactness | native vs generic q8 read, K+V: 32K dmean +.00153 / top -0.52 pt (192 rows, 3 docs), 98K +.00179 / -1.56 pt (64 rows, 1 doc). K-only: 32K +.00197 / -0.78 (2 docs), 98K +.00137 / 0.00 (1 doc). Not bit-exact vs the generic path; size is at or below the Ap reorder floor (+.0026..+.0033). Cut before doc 3, so the 98K CI is missing | strict gate FAIL; thin |
| Q8N slot restore | 98,023 tokens restored, prompt_n 4 (no re-prefill), 16 decoded tokens identical to the fresh-prefill run | PASS |
| Q8N two-slot deep ABBA x3 (98K+90K) | F16 34.033 ms/tok, 23,047 MiB. K-only q8: 32.931 ms/tok, -3.24% (CI95 -1.53..-0.67 ms), 22,235 MiB. K+V q8: +0.05% (CI spans 0), 21,527 MiB (-1,520) | K-only is a real speed gain; quality still fails (Q8S) |
| Q8N nvprof 98K | ofa kernel ms/step: F 7.253, K+V 6.833, K-only 6.447; regs 118/127/121 | done |
| Q8N single 98K / append 4K (K+V) | +0.81% / -0.34%, 2 cycles | indicative |
| N3 replay (4 arms x 5 blocks, Williams) | makespan b3 211.23 s. cap1024 -0.04% (CI +/-0.38), max token gap -48.6% (16.06 -> 8.26 s): predeclared verdict WORTH IT. GDN4 +1.40% makespan (BUNDLE3 is the faster lib in the end-to-end replay). ubatch 512/1024/2048: 107.59 / 103.69 / 102.17 s, VRAM 22,905 / 23,109 / 23,519 MiB. Context overflow (99,876 > 98,304): clean HTTP 400 in all arms, server stays healthy | done |
| N3 "gemm" arm | DEFECT: the arm used the BUNDLE3 lib, which has no `GGML_CUDA_FA_MAXWELL_GEMM` path (only the Codex v4 lib has it). The arm is effectively a second b3 run: -0.06% +/-0.23 is an A-A noise floor, not a GEMM result | GEMM makespan NOT measured |
| N4 GEMM vs stock FA, per-op vs FP64 | 3 layers x 6 stops. Short (4096/4160): stock is closer to FP64 (G/S RMSE ratio 1.19-2.12). Deep (32K-98K): GEMM is closer, ratio 0.05-0.95 (98,048 L39: 4.86e-07 vs 9.03e-06). In-situ dumps match the stock driver exactly | done |
| Stock upstream b12318 | 2-slot deep 37.83 tok/s total (BUNDLE3 F16 here: ~58.9), prefill 90K 147 tok/s, append 4K 100.8 tok/s prompt, peak 22,987 MiB | baseline |

Readings
- N4 changes how the GEMM gate fails: at depth the stock tile, not GEMM, carries most of the error vs FP64. B-vs-A KL measures distance from a noisier reference. A gate scored against an F32/FP64 reference would treat GEMM differently. This is for the owner to decide; the gate is not relaxed here.
- K-only q8 is the first KV change with a clear deep two-slot speed gain (-3.24%, above the 1% bar) and -812 MiB. Its quality (Q8S K-only +.0009..+.0022) fails the strict gate at all depths.
- Native q8 is not exact against the generic q8 read. The difference is within the reorder floor, but the gate asks for exactness, so it fails.
- cap1024 costs no makespan and halves the worst token gap: still the owner's decision.

Still open: Q8N exactness doc 3 and the 98K CI; N5 (b12318 runtime smoke is covered by the stock baseline above, not as an ABBA).

## N3B: GEMM makespan with the v4 lib (14:28-14:53 CET; subdir `day13/N3B/`)
This reruns the void N3 "gemm" arm with a lib that has the GEMM path. GEMM arm: Codex integration-v4 lib (sha 0c7ffbde...) + `GGML_CUDA_FA_MAXWELL_GEMM=1`. b3 arm: BUNDLE3 (sha 94c777dc...). Short profile: 2 agents, 4 turns, 84K -> ~95.8K, ub 1024, cap 0. 3 paired blocks, counterbalanced (b3/gemm, gemm/b3, b3/gemm).

| item | b3 | gemm | change |
|---|---|---|---|
| makespan, mean of 3 blocks | 210.72 s | 177.67 s | **-15.69%** (CI95 -15.88..-15.49) |
| server prefill seconds | 201.75 | 164.33 | -18.5% |
| 1K append prompt_ms, both slots | 14,853 | 12,741 | -14.2% |
| 4K append prompt_ms, both slots | 35,585 | 28,342 | -20.4% |
| peak VRAM | 23,110 MiB | 23,459 MiB | +349 MiB |

Engagement: each GEMM run logs `maxwell_gemm: version=4 ...`, and the preflight refuses a lib without the GEMM path. nvprof GEMM session: sgemm is 29.2% of GPU time and masked_softmax 7.3%. flash_attn_tile still has 80 launches and 11.8% of GPU time, so part of the attention stays on the stock tile. A likely cause is the `ne[3] == 1` guard when both slots batch together; this is not yet confirmed (WP0 counters in WRAPUP).

Readings
- First end-to-end GEMM number on the target workload: -15.7%, with very little spread between blocks (+/-0.2 s). The wrap-up's -20..-25% estimate assumes full engagement.
- The GEMM arm is the v4 lib, not BUNDLE3 + GEMM. It differs from b3 in more than the flag, so the -15.7% belongs to that lib as a whole. WP0 (one lib on BUNDLE3 + v4) separates the two.
- Measurement only. Not adopted: N4 vs FP64 favours GEMM at depth, but GEMM fails the stock-relative KLD gate; the owner decides.
