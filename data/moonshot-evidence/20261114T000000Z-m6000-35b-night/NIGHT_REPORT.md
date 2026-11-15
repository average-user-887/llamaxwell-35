# Night report 14-15 Nov (M6000 35B)

## Outcome
- **Prod = batch8** (swapped ~05:00 15 Nov): stack7 + moe_v3 + `-ub 2048`, cap1024, F16 KV, 2 slots, prefill-single-stream, fused Maxwell FA GEMM.
- N3 replay: original prod 177.5 s -> stack7 ~161.3 s (-9.3%) -> batch8 ~159.9 s. Batch vs stack7: -1.19% (card 16, 5 runs each) and -0.86% (card 24, 5 runs each, ranges do not overlap), about -1.0% combined. Max gap 5.6 s, VRAM +408 MiB.
- Objective "<=156 s" **not reached**. Best gated stack is batch8.
- Reformed gate: KLD (7e, 17) indistinguishable from the reorder floor, moe_v3 bit-identical on those buckets; ctxfull 2x96K alloc OK (card 19).

## Rollback
`docker stop llamacpp-m6000; docker rename llamacpp-m6000 llamacpp-m6000-batch8-failed; docker rename llamacpp-m6000-stack7-prev llamacpp-m6000; docker start llamacpp-m6000`

## Dropped / no gain (all measured)
| idea | result |
|---|---|
| ragged OFA (14) | decode +14% only on both_mixed, N3 neutral; off by default |
| gap bound (15) | bounding the gap erases the gain |
| GDN18 (18) | bit-equal, N3 -0.06%, no gain |
| cap/ub sweep (21b) | ub1536 +1.4%, ub3072 -0.04%, cap768 +2.0%, cap1536 +0.4% (gap 8.1 s) |
| flag variants (23) | no single-stream, -b 4096, ckpt 8/4096: +0.06 / +0.13 / +0.26%, noise |
| cap fine sweep (26) | 896 +2.9%, 1152 +0.2%, 1280 +0.2% |
| dec2, attention rewrite | stopped earlier (attention at 87% bandwidth) |

## What bounds the rest
Phase split (22): of 160 s, slot0 prefill/decode/idle 64/87/9 s, slot1 72/60/28 s; the 5.6 s max gap is one slot frozen while the other prefills a 4K append. Realistic recoverable 3-8 s, unmeasured. Next lever: server-side append scheduling / prefill-decode overlap (needs a design card; bounding the gap did not help).

## Late cards
- Card 25: noise floor A vs A2 (identical config) is 0.17%; fused FA GEMM off costs +5.5% (168.3 s), so that kernel is worth ~5%.
- Card 26: cap896 +2.9%, cap1152 +0.2% (gap 6.2 s), cap1280 +0.2% (gap 6.9 s); cap1024 stays the optimum.
- Card 23/24/21b noise is within ~0.2%, so the -0.9..-1.2% batch gain is real but modest.

## Pointers
Per-card results in `occamy35-night-20261114/<card>/RESULT.md`, decision log `DECISIONS.md`, plots `plots2/`. Code branches night14-ragged14, night14-gap15, night14-gdn18 pushed.

## Day window 15 Nov (16:45-20:00): prefill/decode lever hunt, no new adopt
Prod stays batch8 (159.9 s). All rows below are measured; none passed.
| card | idea | result |
|---|---|---|
| 27 | append cache re-evaluation | DEAD: 18 runs x 2 slots, appended = evaluated (0.00% excess) |
| 28 | prefill per-op profile (nvprof, 94K single slot) | attention 46% at 30K depth, 71% at 92K; MoE mmq 25% -> 13%; dense 21% -> 12%; GDN <=6%. QK 69% / PV 74% of FP32 peak, softmax 78% of BW. QK layout/algo variants: bit-identical, no gain |
| 28b | 2-stream overlap of softmax+PV with next QK tile | +1% of deep attention, bit-identical; cuBLAS fills all SMs |
| 28c | fused exp-on-load PV kernel | fused/current 0.92-0.99 (need <=0.90); kernel 82-97% of cuBLAS sgemm; saves 0.75 ms P round-trip, loses ~1.0 ms GEMM at 92K |
| 28d | MoE/dense efficiency, 2-slot decode profile at 90K | prefill MoE mmq 42-43% of FP32 peak, dense 70%. Decode 32.3 tok/s/slot, 29.6 ms GPU of 31 ms step: attention 44%, MoE 27%, GDN 20%, dense 7.7% |
| 28e | prefill MoE mmq via FP32 dequant SGEMM | 1.09x (gate 1.25x); CUDA-C FFMA loop tops at ~3.3 of 6.8 TFLOPS, needs hand-scheduled SASS; accuracy fine |
| 28f | decode bytes vs 317 GB/s | attention 89% (wall), moe_v3 44%, gdn_fused2 14-45%, dense 62%; ~7 ms/step theoretical |
| 28g | moe_v3 decode occupancy | 1.00x; grid 24 = grid 48; in-phase 200-230 GB/s, fixed costs (barriers, ramp, tail) explain the average |
| 28h | gdn_fused2 decode (A2ROW: two weight rows per activation load) | op 1.131x (gate 1.15x), bit-identical, default off, branch night15-gdn28h; phase A was LDS-bound (100 -> 77 us); phase B serial chain is the next lever. N3 ABBA (2 runs/arm): 159.11 -> 158.25 s = -0.54%, 4/4 faster, bit-equal; below the 1% gate, kept as default-off batch candidate |
| 30 | host flags (threads, poll, prio) | A 159.47 s; B +0.17%, C -0.16%, D -0.14%: noise, dead |
| 29 | V/K Q8 KV | not feasible: fused FA GEMM needs F16 K/V (K-only Q8 was never adopted) |
Reading: every remaining lever needs hand-scheduled SASS (prefill MoE/attention GEMMs) or a new fused kernel; CUDA-C reaches 82-90% of cuBLAS at best. Decode attention is at the bandwidth wall (89%); only fewer KV bytes help, which needs a pack-step dequant for the fused GEMM.
Next candidates (unmeasured): shared-expert rows in moe_v3 phase A idle wave and a persistent two-token launch (~5-10% of the 7.9 ms op each); GDN decode stall check (28h); MaxAs-style SASS PV/QK or mmq core; pack-step q8 KV dequant.
