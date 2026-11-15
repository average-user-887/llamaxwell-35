# Card 28g: moe_v3 decode parallelism. Result: GATE MISSED, nothing integrated

| moe_v3 grid (blocks) | blocks/SM | us/launch (2 reps) | vs prod grid 48 |
|---|---|---|---|
| 48 (prod) | 2 | 93.11 / 93.18 | 1.00x |
| 40 | 2 (partial) | 100.80 / 100.77 | 0.92x |
| 36 | 2 (partial) | 105.58 / 105.56 | 0.88x |
| 32 | 2 (partial) | 114.11 / 114.10 | 0.82x |
| 24 | 1 | 94.05 / 94.09 | 0.99x |

Gate: >=1.15x on the op (<=81 us/launch). Best reached: 1.00x (prod config). Integration and the N3 row were skipped.

## Phase breakdown at grid 48 (globaltimer stamps, us from the first block start, min/max over blocks)

| phase | window | duration | effective BW |
|---|---|---|---|
| A: norm + router (2.1 MB) | 0 -> 10.2/12.3 | ~12 | ~170 GB/s |
| barrier 1 | -> 14.3 | ~2 | |
| B: gate/up rows (10.6 MB) | 14.3 -> 55.3/61.4 | ~46 | ~230 GB/s |
| barrier 2 | -> 63.5 | ~2 | |
| C: down rows (5.3 MB) | 63.5 -> 88.1/90.1 | ~26 | ~207 GB/s |
| launch gap / prologue | | ~3-5 | |

## Finding
- moe_v3 is not occupancy or latency bound. With grid 24 (1 block/SM, half the resident warps) it runs within 1% of grid 48. Grids that leave SMs unevenly loaded are slower because of the tail.
- Within each phase the kernel streams at 200-230 GB/s on scattered 1152 B / 288 B Q4_0 rows. The 141 GB/s figure from 28f averages over the whole launch, so it includes the fixed costs: phase-A ramp, two grid barriers (~4 us), launch gap and block-end spread (~6 us in B).
- More warps, split-K or split rows cannot reach 1.15x. Realistic remaining levers, each worth about 5-10% and untested:
  1. Move the shared-expert gate/up rows (no routing dependency, ~1.2 MB) into phase A's idle second wave. This should be bit-identical.
  2. Run both decode tokens in one persistent launch, which overlaps phases and saves one launch gap. This needs dispatch changes.

## Method
Standalone microbench (bench/mb.cu): the moe_v3 source from occamy-fused.cu at d4c13eb, real decode shapes (H=2048, 256 experts, top-8, ff=512, shared ff=512, Q4_0), synthetic valid Q4_0 weights, one token per launch as in prod, 256 timed launches per rep. Instrumented copy: bench/kern_ts.inc. The absolute time matches the prod trace (91-104 us). Ran under the GPU lock; prod was unloaded through the model's unload endpoint.
