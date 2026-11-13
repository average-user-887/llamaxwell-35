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
