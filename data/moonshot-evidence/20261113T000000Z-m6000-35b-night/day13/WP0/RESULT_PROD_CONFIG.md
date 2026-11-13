# WP0 result: candidate = WP0 build (2436f75 + cap patch + GEMM v4 + counters + KVQ8)

Arms: b3 = G1b binary + bundle3 lib, F16 KV, ub1024, cap0 (prod-equivalent). gemm = WP0 build, GEMM on, F16 KV, ub1024, cap0. full = WP0 build, GEMM on, K q8_0 / V f16 (KVQ8), ub2048, cap1024 (adoption candidate). Short profile.

Runs ok: b0_p0_b3=True, b0_p1_prod=True, b1_p0_prod=True, b1_p1_b3=True, single_prod_s0=True, single_prod_s1=True

## Single slot: +4K append at 90K (restore 84K, append to 90K, then 90K->94K)

| run | n_restored | 84K->90K prompt_ms | 90K->94K prompt_ms | wall s | engaged | fallback | reasons | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|
| single_prod_s0 | 84000 | 43451 | 30403 | 30.43 | 0 | 90 | q_lt_64:40,q_gt_1024:50 | 23384 |
| single_prod_s1 | 84000 | 43697 | 30542 | 30.57 | 0 | 90 | q_lt_64:40,q_gt_1024:50 | 23384 |

Accept (WP0): gemm single +4K at 90K within 23.4 s +-3% (22.70..24.10 s) and engaged > 0.

## Two-slot N3 makespan (s), paired per block

| block | b3 | gemm | full | prod | gemm vs b3 | full vs b3 | prod vs b3 |
|---|---|---|---|---|---|---|---|
| 0 | 210.59 | - | - | 184.92 | - | - | -12.19% |
| 1 | 210.67 | - | - | 184.32 | - | - | -12.51% |
- prod vs b3: paired mean -12.35% (CI95 -14.36..-10.34, n=2)
- reference (N3B, 3 blocks, same short profile): b3 210.72 s, whole-v4 lib 177.67 s (-15.69%)

## Per arm means (main)

| arm | makespan s | prefill s (server) | decode s | max gap s | peak VRAM MiB | headroom MiB (of 24576) |
|---|---|---|---|---|---|---|
| b3 | 210.63 | 201.71 | 159.24 | 16.05 | 23109 | 1467 |
| prod | 184.62 | 182.36 | 140.62 | 6.05 | 23867 | 709 |

## GEMM counters per run (last occamy_fa_gemm line, printed at backend free)

| run | engaged | fallback | reasons |
|---|---|---|---|
| b0_p1_prod | 120 | 19370 | ne3:11680,q_lt_64:7650,q_gt_1024:40 |
| b1_p0_prod | 120 | 19370 | ne3:11680,q_lt_64:7650,q_gt_1024:40 |
| single_prod_s0 | 0 | 90 | q_lt_64:40,q_gt_1024:50 |
| single_prod_s1 | 0 | 90 | q_lt_64:40,q_gt_1024:50 |

## Q8 slot set (full arm): restore count + round trip (16 tokens decoded twice after erase+restore)

- n3_slots_84000.json: n_restored=[84000, 84000], build_s=722, roundtrip=[{'slot': 0, 'identical': True, 'n': 16}, {'slot': 1, 'identical': True, 'n': 16}]

## Receipts

- b0_p0_b3: 8/8 receipt_ok; b0_p1_prod: 8/8 receipt_ok; b1_p0_prod: 8/8 receipt_ok; b1_p1_b3: 8/8 receipt_ok
