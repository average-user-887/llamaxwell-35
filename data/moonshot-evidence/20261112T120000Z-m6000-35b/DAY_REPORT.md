# M6000 35B (llamaxwell-35) day report, 12 Nov 2026

Lead: Claude Opus session (lab authority from owner), directed by Fable advisor session. Night planning: owner's auditor.
Rule today (advisor): measurement only. No adoptions, no prod changes, no pushes to main; evidence goes on branch `evidence-20261112` only.

## Prod state (unchanged all day)
- Container llamacpp-m6000 runs BUNDLE3: lib sha 94c777dc…, GH main source 2436f75, `-np 2`, F16 KV, GEMM off, graphs off.
- Rollback container llamacpp-m6000-gdn4-prev is kept.
- Overnight lineage 11->12 Nov: BUNDLE2 adopted 04:05, then GDN4 at 08:30 (+0.86% warm, +0.82% d4k, bitexact, +135 MiB), then BUNDLE3 (N2SPLIT + FA_ADDR + N2GDN).

## Results today

| id | what | result | status |
|---|---|---|---|
| E1 | BUNDLE3 flag engagement | N2SPLIT, FA_ADDR and N2GDN confirmed by kernel names per decode step | done |
| E4 | phase profile BUNDLE3 (B) vs GDN4 (A), single run | N2 deep 36.79 -> 33.38 ms/tok (-9.3%); N2 shallow 21.77 -> 18.62 (-14.5%); N1 13.42 -> 12.99 (-3.2%); prefill4K flat. B deep busy: ofa_decode_hs_ar 45% (14.24 ms), moe_v2 8.57, gdn_fused2 5.85, mmvq 2.25 | indicative, not ABBA |
| C1 | BUNDLE3 2-slot ABBA | N2SPLIT -4.52%, N2GDN -3.33%, B3 total -8.98% (common-window 54.15 -> 59.07 tok/s) | PROVISIONAL: 1 ABBA block |
| C2 | 2-slot numerics | BUNDLE3 vs solo KL .00089; stock pair vs solo .00306 | done |
| C3/C4 | provenance SHAs; lifecycle cases | recorded; tool-level lifecycle bit-equal | done |
| Q8 | KV capacity sizing at 2x128K | F16 5,120 MiB = no-go (~75 MiB margin); q8 K+V 2,720 MiB (~2.5 GiB margin); needs native q8 read in ofa_decode_hs_ar | sizing only; E10 quality NOT_RUN |
| S1 | Codex GEMM prefill 65K KL outlier (source audit) | no bug found; the outliers fall on n_kv tail-0 windows | hypothesis only |
| G1 | GEMM prefill numerics, row 0, n~5/bucket | A-vs-B up to .294 at 65K, but stock tile-order change alone gives .259; controls bitwise deterministic | measurement; noise-dominated |
| G1b | scheduler cap `--mixed-batch-size 1024` vs 0, GEMM off, 5 cycles | max visible gap -48% (90K append) / -45% (deep+16K); TTFT +0.6% / +2.2%; wall +0.4% / +1.5%; VRAM flat | candidate for owner decision; not adopted |
| E9 | deep single-slot decode at 98.1K, `-lv 4` cost; gdb sampling | 19.025 vs 19.073 ms/tok (+0.25%); host in cudaStreamSynchronize 97.5% of decode samples, so GPU-bound | done |
| G1c | GEMM multi-row gate (8 docs x 64 rows at 32K/65K/98K) + FGDIV | B-vs-A dmean +.00345/+.00253/+.00332, same-top -1.17/-0.78/-0.98 pt. 32K/65K comparable to the doc-0 A-vs-Ap floor (+.00447/+.00119); 98K is about 2x the floor (+.00168). FGDIV cap0 vs cap1024: mean KL .0059-.0072, same-top 100%, spread over decode rows rather than coming from prefill | FAIL in every bucket. 32K/65K: gate not meaningful at this depth (owner decision). 98K: plain fail. Noise floor rests on 1 doc |
| E5 / E7 | MoE routing capture; upstream b12318 sm_52 smoke | not run today | queued |

## Advisor rulings
- GEMM prefill (Codex, prefill -22..-27%, +412 MiB) is NOT adoptable on current evidence. G1c decides it, using the fixed owner gate: B-vs-A dmean <= +.0010 and same-top >= -.3 pt in every bucket. A failure that matches the A-vs-Ap noise floor is reported to the owner as "gate not meaningful at this depth"; the gate is never relaxed.
- Cap1024 goes to the owner for a decision. G1c FGDIV explains the cap0-vs-cap1024 fg greedy divergence. CPU result so far: workload a timed streams are identical, the identity pass diverges at token 26 of 32, and workload b is confounded by its prompt rotation.
- 16-row same-top1 in G1: B was 1 pt lower than A against the F32 reference at 16K and 32K (96 rows each), equal at 65K (80 rows).

## Findings worth carrying into the night
1. Deep decode is GPU-bound (E9). Host-side work (graph build, HOSTGAP) has little headroom left at 98K. The kernel target is ofa_decode_hs_ar, 45% of deep busy (E4).
2. The 2x128K context goal needs q8 KV (Q8 sizing). Native scope: q8 loads in ofa_decode_hs_ar plus prefill dequant. The E10 quality screen comes first.
3. The n_kv tail is PAD256(max_pos+1) (G1c/TAIL.md, llama-kv-cache.cpp:1280-1292). G1's tail tags were right only for row 0.
4. The SSE 510/512 IDs are UTF-8 holdback in server streaming and content-dependent. The authoritative receipt is `tokens_predicted`.
5. GGML_CUDA_MAXWELL_GRAPHS must stay unset: any value, including =0, enables graphs and crashes BUNDLE3.
6. C1 needs >= 5 ABBA blocks before the -8.98% figure is final (E2).

## Open / NOT_RUN
- E2: BUNDLE3 ABBA with >= 5 blocks.
- Full BUNDLE3 2-slot logits vs GDN4.
- Transitions, slot swap, cancel/restore.
- E10: Q8 KV quality screen.
- E5: routing capture.
- E7: b12318 runtime smoke (build exists, unsmoked).
- E3: arm B.
- G1c follow-up: a multi-doc noise floor (A-vs-Ap on more than one doc) and the first divergent index per FGDIV unit.

## Where things are
- Branch `evidence-20261112`, dir `data/moonshot-evidence/20261112T120000Z-m6000-35b/`. Subdirs: G1, G1b, G1c, E4, E5, E9, Q8, S1, raw/C, harness, historical. summary.json is the machine index.
- Paths redacted: `<LAB>` is the lab storage root, `<MODELS>` the models root, `<lab-endpoint>` and `<gpu>` replace endpoint and device ids.
