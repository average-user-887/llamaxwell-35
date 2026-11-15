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
Per-card results in `occamy35-night-20261114/<card>/RESULT.md`, decision log `DECISIONS.md`, plots `plots2/`. Code branches not yet pushed at writing: night14-ragged14, night14-gap15, night14-gdn18.
