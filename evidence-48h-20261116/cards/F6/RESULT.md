# F6 leave-one-out of bundle9 (N3, n=6 per arm, palindromic X Y Z Z Y X x3)

| arm | makespan mean s | sd | delta vs control s | % | engagement |
|---|---|---|---|---|---|
| X control (all flags) | 158.56 | 0.10 | +0.00 | +0.00% | ragged guard 1 line, fallbacks=8192, gdn18 33000 |
| Y minus RAGGED(+ratio) | 158.50 | 0.18 | -0.06 | -0.03% | no ragged lines, gdn18 33000 |
| Z minus GDN18 | 158.53 | 0.22 | -0.03 | -0.02% | ragged guard fallbacks=8192, gdn18 absent |

All 18 runs valid (rc=0, nonzero counters where expected). A2ROW on in all arms. Reference R0: batch8 159.26 s, bundle9 158.75 s.

Method: F6/f6_gpu.sh derived from R0/gpu_r0.sh (flags/env, arm loop, lock 43200 s, prefix cleanup only); R0 build libs for all arms; N3 prod order, cap1024 ub2048 b2048, batch8 env base. Lock held 06:54-07:52, prod unloaded by POST.

Caveats: control here is 158.56 s, within noise of every leave-one-out arm (sd 0.1-0.2 s, deltas <=0.06 s) so neither flag measurably carries the gain; the -0.32% vs batch8 is not attributable to RAGGED or GDN18 individually at this n (A2ROW and the build are the remaining candidates, untested). Ragged guard: the only logged decision is "-> stock split" (to_ragged=0, fallbacks=8192), so RAGGED probably never engages in N3 (ratio 2 guard falls back); engagement counter "ragged on" absent in log. Harness names containers n28h-* (cannot be changed without code edits), so cleanup removes only n28h-* containers absent at lock time plus nf6-*. 250 W untouched, MAXWELL_GRAPHS unset.

Files: F6/f6_gpu.sh, F6/run-20261116T045454Z/ (steps.log, per-run N3_TABLE.md), F6/launch.log
