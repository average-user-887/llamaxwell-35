# WP0 result: candidate = WP0 build (2436f75 + cap patch + GEMM v4 + counters + KVQ8)

Arms: b3 = G1b binary + bundle3 lib, F16 KV, ub1024, cap0 (prod-equivalent). gemm = WP0 build, GEMM on, F16 KV, ub1024, cap0. full = WP0 build, GEMM on, K q8_0 / V f16 (KVQ8), ub2048, cap1024 (adoption candidate). Short profile.

Runs ok: b0_p0_b3=True, b0_p1_gemm=True, b0_p2_full=False, single_b3_s0=True, single_full_s0=True, single_full_s1=True, single_gemm_s0=True
ERRORS: [('b0_p2_full', ' GEMM_NOT_ENGAGED')]

## Single slot: +4K append at 90K (restore 84K, append to 90K, then 90K->94K)

| run | n_restored | 84K->90K prompt_ms | 90K->94K prompt_ms | wall s | engaged | fallback | reasons | peak VRAM MiB |
|---|---|---|---|---|---|---|---|---|
| single_b3_s0 | 84000 | 44669 | 31248 | 31.27 | - | - | - | 22974 |
| single_full_s0 | 84000 | 43602 | 30468 | 30.49 | 0 | 90 | ktype:90 | 22484 |
| single_full_s1 | 84000 | 43686 | 30470 | 30.50 | 0 | 90 | ktype:90 | 22484 |
| single_gemm_s0 | 84000 | 33168 | 23091 | 23.12 | 100 | 40 | q_lt_64:40 | 23324 |

Accept (WP0): gemm single +4K at 90K within 23.4 s +-3% (22.70..24.10 s) and engaged > 0.

## Two-slot N3 makespan (s), paired per block

| block | b3 | gemm | full | gemm vs b3 | full vs b3 |
|---|---|---|---|---|---|
| 0 | 210.68 | 177.48 | 209.70 | -15.76% | -0.46% |
- gemm vs b3: -15.76% (n=1)
- full vs b3: -0.46% (n=1)
- reference (N3B, 3 blocks, same short profile): b3 210.72 s, whole-v4 lib 177.67 s (-15.69%)

## Per arm means (main)

| arm | makespan s | prefill s (server) | decode s | max gap s | peak VRAM MiB | headroom MiB (of 24576) |
|---|---|---|---|---|---|---|
| b3 | 210.68 | 201.78 | 159.22 | 16.06 | 23109 | 1467 |
| gemm | 177.48 | 163.94 | 140.74 | 11.81 | 23459 | 1117 |
| full | 209.70 | 213.94 | 157.20 | 8.39 | 22619 | 1957 |

## GEMM counters per run (last occamy_fa_gemm line, printed at backend free)

| run | engaged | fallback | reasons |
|---|---|---|---|
| b0_p1_gemm | 160 | 19260 | ne3:11710,q_lt_64:7550 |
| b0_p2_full | 0 | 19530 | ktype:19530 |
| single_full_s0 | 0 | 90 | ktype:90 |
| single_full_s1 | 0 | 90 | ktype:90 |
| single_gemm_s0 | 100 | 40 | q_lt_64:40 |

## Q8 slot set (full arm): restore count + round trip (16 tokens decoded twice after erase+restore)

- n3_slots_84000.json: n_restored=[84000, 84000], build_s=722, roundtrip=[{'slot': 0, 'identical': True, 'n': 16}, {'slot': 1, 'identical': True, 'n': 16}]

## Receipts

- b0_p0_b3: 8/8 receipt_ok; b0_p1_gemm: 8/8 receipt_ok; b0_p2_full: 8/8 receipt_ok
