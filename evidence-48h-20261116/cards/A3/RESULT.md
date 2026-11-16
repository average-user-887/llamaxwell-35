# A3 - GDN phase-B seq0/seq1 chain interleave

| item | value |
|---|---|
| commit | 5df383c on a3-gdn-il (base 52c9e84, bundle9) |
| libggml-cuda.so sha256 | bf46e0a5cfe0b751e94a5b97d7b2053973824d15b1d5e4bf7fba57d89b986ea9 |
| regs / spill (IL0..IL3) | 80 / 0 all; static smem 1392 / 2784 / 2912 / 2912 B |
| op IL0 A2ROW base us/launch | 172.24 (run1), 172.98 (run2) |
| op IL1 card18 us / x | 177.24 / 0.972x; 176.69 / 0.979x |
| op IL2 A3=1 us / x (gate >=1.08x) | 176.02 / 0.979x; 176.10 / 0.982x -> FAIL |
| op IL3 A3=2 (+next-item prefetch) us / x | 175.59 / 0.981x; 176.64 / 0.979x -> FAIL |
| bitwise mb IL1/IL2/IL3 vs IL0 (out conv ssm) | all bit-equal, diff 0, both runs -> PASS |
| server CHECK=64 bitequal=0 count | 0 of 64 lines -> PASS |
| N3 short ABBA (A=bundle9 IL1, B=A3 mode 2/IL3) | makespan A 159.30/158.55 (mean 158.93 s), B 158.81/158.65 (mean 158.73 s): B -0.12% (-0.20 s), within A-A spread 0.75 s = noise; engaged yes (a3 engaged launches 33000, IL=3) |
| RMSE vs FP64 | = base if bitwise holds (same expressions, same reduction order) |

Flag: `GGML_CUDA_OCCAMY_GDN_A3=1` (IL2) or `=2` (IL3), needs n2gdn path; overrides GDN18. Default off.
Debug: `occamy-gdn: a3|gdn18 variant IL=.. blocks/SM`, `a3 engaged launches N (IL=..)`, `a3 check ... bitequal=`.
Change: 4 chains (2 seq x 2 col) share one interleaved xor butterfly for k.S and q.S'; ssq partials parity
double-buffered in smem -> 2 barriers per item instead of 4. IL3 also prefetches next item's state tiles.

Run: `N3_GPU=<gpu> N3_UNLOAD_URL=<lab-endpoint>/api/models/unload/qwen36-35b-moe A3/a3_gpu.sh` (~20 min, lock -w 43200).

Risks: card 18 IL (same idea, per item) was bit-equal but N3 -0.06%; 28h next-item prefetch gave no gain;
1.08x needs phase B (~46 of 177 us) to lose ~13 us, near the 8.4 MB state traffic floor (~28 us whole op).

## GPU run (A3 gpu)
Verdict: bitwise PASS, engagement yes, speed gate FAIL (best 0.982x vs >=1.08x); N3 delta -0.12% noise. Run dir: A3/run-20261116T034132Z (steps.log, mb/run1.log, mb/run2.log, check.log, A_1/B_2/B_3/A_4). Method: a3_gpu.sh unmodified, flock queue, N3 short profile ABBA x1. Caveat: single ABBA, mode picked by IL3>IL2 (0.981 vs 0.979, near-tie).
