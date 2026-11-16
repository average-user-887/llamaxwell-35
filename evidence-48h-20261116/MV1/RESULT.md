# MV1a: why 35B MTP verification is slower than no-spec (M6000, sm_52)

Status: GPU run finished 16 Nov (run id 20261116T163133Z, 19 arms, all ok=1, no OOM/crash). Runtime verdicts filled below from node timing + host timers of the trace arm and the paired throughput arms. Numbers below marked (M2b) are prior measurements, not new ones.

## Table 1: verdict per component (code trace + runtime verdicts)

| Component | What verify (k+1=2 rows) hits | Source | Cost evidence (M2b, 80K, 1 slot) | Runtime verdict |
|---|---|---|---|---|
| FA, target 10 layers, F16 KV | maxwell_gemm needs Q rows >= 64 -> skip; occamy split-K needs Q->ne[1]==1 -> skip; falls to stock TILE (cc 5.2, GQA) | fattn.cu:786-817 (dispatch), fattn-maxwell-gemm.cuh:356 (rows>=64), fattn-occamy.cu:811 (ne[1]!=1), fattn.cu:~708-730 (tile choice) | stock tile 16->22 ms vs 6 ms native | PARTLY CONFIRMED. mv1_fa counters were NOT obtained (0 lines; dump is atexit/20000-call and the container is force-removed). Node timing, trace arm: verify graph (dec2, 127 graphs) FLASH_ATTN_EXT 16.35 ms/graph = 1.63 ms/layer x10, ~2.7x the ~0.6 ms/layer native 1-row cost (M2b 6 ms/10 layers). Consistent with stock tile, not proven |
| FA, MTP head, q8 draft KV | occamy kvq8 path also Q->ne[1]==1 only and needs GGML_CUDA_OCCAMY_KVQ8=1 (not set) -> stock | fattn-occamy.cu:854-873, 869 | part of head 16-21 ms | Head FA 7.7 ms/call (274 calls, trace arm), stock path assumed (counters missing); head total 17.1 ms/round incl. re-eval |
| MTP head re-eval | server re-evaluates draft ctx on the whole verify batch every round (TAG_SPEC_AVOID_DRAFT_REEVAL) in addition to the 1-row draft | server-context.cpp ~4276 (common_speculative_process), 3413 (draft), speculative.cpp ~1540-1640 | head 16-21 ms/step | CONFIRMED cost: draft-ctx re-eval 8.9 ms/round + 1-row head draft 8.2 ms/round = 17.1 ms/round (host timers, delta rounds 50->100). Head FA alone 7.7 ms/call (274 calls, q8 draft KV, stock path) |
| GDN (recurrent) | occamy gdn window matcher is 1-token-per-seq decode shaped (ssm_conv ne[1]==1, n_seqs match); verify graphs are multi-token-per-seq -> stock GATED_DELTA_NET with snapshot slots; windows rejected in M2b dbg log | occamy-gdn.cu:943,948,965,1154; ggml-cuda.cu:3030-3060 | non-attn target+head 32.1 ms (k1) vs ~12.5 (k0) | Not separable: GDN fused node only 1.45 ms/graph (dec2); GDN is NOT the bottleneck (M2b non-attn estimate was too high) |
| MoE mul_mat_id / dense mmq / mmvq | no occamy n-row variant; stock mul_mat_vec_q / mmq used; fused windows rejected (unexpected mul_mat/SCALE/ROPE) | ggml-cuda.cu fused-window matcher; M2b dbg | mul_mat_vec_q 6-9 ms | Verify graph dec2 has a fused 'RMS_NORM@MoE' window costing 8.0 ms/graph (largest non-FA item, includes expert matmuls); MUL_MAT_ID in the 1-row head 3 x ~1.0 ms. No finer split available |
| Rollback/ckpt | target n_rs_seq=2, draft ctx n_rs_seq=0; with draft<=2 expected no ckpt restore (n_rollback <= n_rs_seq) | server-context.cpp 4455-4540, llama-context.cpp:108-112 | expected 0 restores | CONFIRMED 0 restores in 100 rounds (acc_hist [5,95,0,0]: 5 rejections handled without ckpt restore). Target logged n_rs_seq=1 (not 2) |

Hypothesis "verify batches miss the Occamy Q->ne[1]==1 decode path and drop to stock tile": SUPPORTED BY CODE + INDIRECT TIMING, NOT CONFIRMED BY COUNTERS (counter dump never flushed, see Failures). Needs a rerun with flush on a periodic timer or a signal, 1 short trace arm.

## Verdict (measured, A = no-spec, B = MTP draft-n-max 1, ~80K prompt per slot, greedy, ABBA order, prod libs)

| Config | A tok/s | B tok/s | B/A | accept (B) | Speedup from acceptance alone would be |
|---|---|---|---|---|---|
| 1 slot greedy (2 pairs) | 57.7, 58.2 | 37.27, 37.21 | 0.645 | 188/194 = 0.969 | +97% (wrong: real is -35%) |
| 2 slots greedy (3 pairs, per slot, sum) | 33.4+33.7 = 67.0 | 19.6+19.2 = 38.8 | 0.58 | 186/196 = 0.949 and 163/220 = 0.741 | |
| 1 slot sampling (server defaults, seed) | 58.12 | 35.96 | 0.619 | 182/201 = 0.905 | |
| Prompt processing, 1 slot / 2 slots | 305.3 / 153 | 276 / 138 | 0.905 / 0.90 | | |

Repeat spread is <1% (A 57.66-58.16; B 37.21-37.27; np2 B slots 19.57-19.61 / 19.17-19.19), so the gap is far above noise. MTP is a loss in every arm measured. Acceptance is high (0.90-0.97), so this is a step-cost problem, not an acceptance problem.

## Bottleneck (trace arm, GGML_CUDA_NODE_TIMING + host timers, 1 slot, 80K; timing overhead lowers tok/s to 33)
Per MTP round (1 draft + 1 verify, 1.97 tokens): total ~53 ms in prod arm (1/37.24 s x 1.969 tokens). Trace decomposition (ms/round): target verify decode 41.1 (dec2 graphs: 39.95 ms/graph; FA 16.35, fused MoE window 8.0, qkv 1.74, GDN 1.45, output 1.33, rest ~11), draft-ctx re-eval of verify batch 8.9, 1-row head draft 8.2 (of which FA 7.7 per call incl. q8 KV), sampling/accept 0.4. No-spec step is 17.4 ms.

## Table 2: break-even (measured, replaces the M2b table)
No-spec step 17.37 ms (57.9 tok/s). With tokens/round 1+a:

| Accept a | Max MTP round for break-even | Measured round (prod, 1 slot) | Gap |
|---|---|---|---|
| 0.97 (observed greedy) | 34.2 ms | ~52.9 ms | -18.7 ms |
| 0.905 (observed sampling) | 33.1 ms | ~53.0 ms (samp_B) | -19.9 ms |
| 1.00 (ceiling for k=1) | 34.7 ms | 52.9 ms | -18.2 ms |

The round must shrink by >=19 ms (36%) just to tie, and by ~24 ms for a 5% gain. Confidence: HIGH that the verify step cost (not acceptance) is the problem; MEDIUM on the per-op split (single trace arm with node-timing overhead, 'RMS_NORM@MoE' fused label is coarse).

## Greedy equivalence and state across rejection
- Output is NOT bit-identical to no-spec: first differing token 26 (1 slot, identical in both B pairs and the trace arm); 2 slots: 18 and 35. Reference for how much divergence the numerics alone cause: no-spec 1 slot vs no-spec 2 slots already diverge at token 44, so the B divergence is the same class (batch-shape numerics in the 2-row verify vs 1-row decode), not evidence of a state bug. B vs B is deterministic across repeats.
- Rejection handling: 5 rejections in 100 rounds, restores=0, no crash/garbage; acceptance histogram [5,95,0,0] (n_max=1).
- Not tested: a rejection-heavy fixture and a full logits-level KLD for B vs A.

## Memory (peak nvidia-smi MiB, 5 s sampling)
| Config | A | B | B-A |
|---|---|---|---|
| 1 slot | 21174 | 22165 | +991 |
| 2 slots | 22882 | 24187 | +1305 |
VRAM fits in every arm (24 GiB card, 2-slot B leaves ~390 MiB). No OOM.

## Table 3: ranked changes (re-ranked on measured ms; none implemented here)

| # | Change | Kind | Est. ms/round saved | Notes |
|---|---|---|---|---|
| 1 | Multi-row (2-4 rows) GQA split-K FA decode path for F16 KV, target 10 layers | NEW KERNEL | 8-11 (16.4 -> ~6-8) | needs mv1_fa counter confirmation first; bit-exactness vs tile |
| 2 | Skip/narrow draft-ctx re-eval of the verify batch | server logic | 4-8 of 8.9 | head KV bookkeeping, numerics must stay |
| 3 | Fused-MoE window for 2-row verify (8.0 ms/graph window) | NEW KERNEL/matcher | 3-5 | split not known, needs finer node timing |
| 4 | q8-KV multi-row FA for the MTP head (7.7 ms/call) | NEW KERNEL | 4-6 | flag GGML_CUDA_OCCAMY_KVQ8 exists for 1 row only; head KV in F16 would cost VRAM |
| 5 | GDN n-row | NEW KERNEL | <1 | demoted: GDN is only 1.45 ms/graph |

Stack 1+2+3+4 ~ 19-30 ms vs 19 needed to tie: at best break-even to a small gain. Recommendation: do not spend more GPU time on MTP k=1 at this context until (1) and (4) exist; no change to prod (batch8).

## Failures / caveats
- mv1_fa per-path counters absent (dump never flushed; container force-removed). Hypothesis status above stays "not confirmed".
- Host timers cover rounds 50 and 100 only (trace arm), tgt_decode includes the 80K prefill in its cumulative value; per-round numbers are deltas.
- 2 slots in B: slot 1 accept 0.74 (different prompt), skews the 2-slot figure.
- No arm failed; steps.log: all 19 arms ok=1, lock released, cleanup done.

## Run artifacts
Raw per-arm summaries (token ids and prompts removed): publish/measurements/ (arms.csv, per-arm result_summary.json, vram.csv, steps.log, trace_B.nt.tsv). Scripts: publish/scripts/ (mv1_summarize.py builds the measurements dir). Patch: publish/patches/.
