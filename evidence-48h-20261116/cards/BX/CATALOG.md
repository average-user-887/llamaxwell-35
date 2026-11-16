# BX catalog: small gains (gain >0 measured, failed gate or <1% alone), 16 Nov 2026
Base = prod batch8 d4c13eb, N3 makespan ~159.3 s, sd ~0.1-0.2 s (noise floor A/A 0.17%).
Counters: GGML_CUDA_OCCAMY_DEBUG=1 stderr lines.

## Included (measured gain > 0)
| id | branch/commit | measured delta (sd) | engagement counter | VRAM delta | token-identical | conflicts |
|---|---|---|---|---|---|---|
| A2ROW (28h) | night15-gdn28h 54efa6d, GGML_CUDA_GDN2_A2ROW=1, NOT in d4c13eb | N3 -0.54% (159.11->158.25 s, n=2/arm, spread <=0.14 s); op 1.131x | "n2gdn variant A2ROW", "n2gdn launches N" (33000) | 0 | yes, bit-equal | contained in bundle9; shares occamy-gdn.cu with GDN18/A3 |
| bundle9 (R0) | bundle9 52c9e84 (= A2ROW + GDN18 + RAGGED + ratio guard + gap15) | N3 -0.32% (158.75 vs 159.26 s, n=6/arm) | all arms engaged | ~0 | numerics-only checked, N3 identity not recorded | superset of A2ROW; F6: removing RAGGED or GDN18 changes nothing (158.50/158.53 vs 158.56), so the gain is A2ROW's, and it is -0.3% not -0.54% |
| cap2048 (arm C, PMAX default) | flag only (--cap 2048), prod libs/bundle9 | -0.16% n=2 on batch8 (earlier -1.65% on stack7 was ub2048) | n/a | not recorded (+1.7% earlier) | not checked | max token gap 5.6 -> 10.8 s (fails gap gate) |

## Pending (results not landed; cannot be counted)
| id | branch/commit | status | notes |
|---|---|---|---|
| H1 ckpt pinned copy | h1-ckpt abf7711, LLAMA_H1_CKPT=1 | A/B n=6 running/queued | expected <=0.5%, bytes identical, 0 MiB, counter H1_CKPT_ENGAGED |
| FA0 knob sweep | fa0-split (env knobs only) | GPU row queued | no code to merge unless a knob >=1.05x |
| D1b graphs | d1b-graphcount a37a3c2, GGML_CUDA_MAXWELL_GRAPHS (test containers only) | queued | counter d1b_g...; crash risk in GDN split2 host sync |

## Excluded
| id | reason |
|---|---|
| ub2048, moe_v3, stack7 | already in prod batch8 |
| A2 (+0.06% noise, op 0.955x), A3 (-0.12% noise, op 0.98x), A4 (+0.23% slower), FK1 (0.917x) | zero/negative or op regression |
| A5 p1-p3, A5b | slower/diverges |
| RAGGED (+0.4% N3), GDN18 (-0.06% noise, F6 null), gap15 PMAX (erases gain), flags card 23/30, cap/ub sweeps 21b/26, ckpt 8/4096 | no gain >0 beyond noise or negative |
| MTP, B4 | not adoptable (cost) |
