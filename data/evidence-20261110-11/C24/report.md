# C24 merge: C10 GDN_QKREUSE + C11 GDN_PCSHARE + C07 MOE_DOWNH=2 (2026-11-11)

Clone cand/C24 from src-pin 71d1e436. C10.diff, C07.diff applied; C11 hand-merged into occamy-gdn.cu
(`template <PHASE,RQK=0,PCS=0>`, dev_state occupancy = min over <0> and engaged variant, 3-way dispatch,
debug "qk_reuse %d, pc_share %d"). REVIEW LOW fixes: C07 `#pragma unroll` (~291), sequenced DOWNH debug
counter (~627), C11 `static_assert` NCG power of two. All flags default OFF.
Lib: cand/C24/build/bin/libggml-cuda.so sha256 8a2a37d4ad9e4ffd0fd2d2766b2bda9e27aeb19e70a66045a90e67bd6d497299.
ALL-ON env: GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 (on C04 base env incl. DOWNX=1).

| Gate | Result |
|---|---|
| G0 build | rc=0, only occamy-fused/occamy-gdn objects rebuilt, no new warnings (raw/build.log) |
| G0 resources | gdn_fused<0,{0,1},{0,1}> 72 regs, 0 spill, <=1392 B static + 36608 B dyn smem; moe_v2<0,5>/<0,13> (DOWNH=2) 93/92 regs, 0 spill. Runtime: gdn grid 48 (2/SM), DOWNH=2 variant 2 blocks/SM -> grid 48 |
| G0 SASS OFF | gdn<0..3> == pin <P,0,0>, moe_v2 10 OFF variants identical (raw/sass/summary.txt) |
| CPU oracles | C10 11 checks 0 failed; C11 13 checks 0 failed incl. 3 negative controls (raw/oracle) |
| G3 | ALL-ON: qk_reuse 1, pc_share 1, grid 48; qk_reuse/pc_share/DOWNH=2 launch lines present; ALL-OFF: 0/0, no engaged launches |
| G1 | 3x ALL-OFF + 3x ALL-ON, 64 teacher-forced steps: BITEQUAL 64/64, max_abs 0; state dumps step64 652476ce477f, step2112 5be5fc2d816d identical in all 6 (= T1 baseline) |
| G4 warm_decode (-n512 -r5, 5 ABBA cycles + warmup, 22/22 valid) | A 79.947 -> B 83.946 tps: **+3.999 tps, +5.003%, CI95 [4.991, 5.014]%**, noise floor 0.0091 tps; GAIN |
| ctx-4k (-d4096 -n128 -r3, 5 cycles) | A 77.646 -> B 81.371 tps: +3.725 tps, +4.798%, CI95 [4.726, 4.870]%, floor 0.054 tps; GAIN |

Stack vs isolated: isolated sum 5.17% (compounded 5.26%); measured 5.00% -> 0.17 pp short, beyond noise.
Pairwise arm (A = C10+C11, B = all-on, warm_decode): A 82.719 -> B 83.931, C07 marginal +1.466% CI95
[1.461, 1.470]% (isolated 1.26%: no loss). So GDN pair alone = +3.47% vs isolated 2.44+1.47 = 3.91%:
the shortfall is overlap between C10 and C11 inside the same gdn_fused kernel. No member negative.
Verdict: merged stack passes all gates, +5.0% warm decode, +4.8% at 4k ctx; clears 1% gate.
