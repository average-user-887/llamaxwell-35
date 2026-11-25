# C1 two-slot native MTP: FINAL (REJECT)


| run | makespan s | pause in timed s | makespan-pause s | tokens | tok/decode-s all | non-void | draft acc | void | tmax | VRAM MiB | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abba3/1_OFF64 | 170.29 | 33.3 | 136.99 | 1920 | 42.8 | 46.1 | 0/0 | 8 | 77 | 22117 |  |
| abba3/3_OFF64 | 169.72 | 35.5 | 134.22 | 1920 | 35.1 | 23.0 | 0/0 | 8 | 77 | 22117 |  |
| abba4/1_OFF78 | 189.25 | 33.3 | 155.95 | 1920 | 32.5 | 27.2 | 0/0 | 8 | 77 | 22709 |  |
| abba4/4_OFF78 | 189.16 | 36.3 | 152.86 | 1920 | 33.7 | 23.2 | 0/0 | 8 | 77 | 22709 |  |
| abba3/2_ON64 | 158.68 | 24.1 | 134.58 | 1412 | 26.7 | 19.1 | 689/711 | 8 | 77 | 23233 | SHORT 1412<1920 early=[(0, 5, 120, 'eos'), (1, 0, 123, 'eos'), (1, 1, 122, 'eos'), (1, 2, 2, 'eos'), (1, 3, 122, 'eos'), (1, 4, 122, 'eos'), (1, 5, 1, 'eos')] |
| abba3/4_ON64 | 171.59 | 30.2 | 141.39 | 1920 | 44.1 | 46.2 | 942/957 | 8 | 77 | 23233 |  |
| abba4/2_ON78 | 193.37 | 35.3 | 158.07 | 1920 | 32.8 | 31.7 | 941/958 | 8 | 77 | 23909 |  |
| abba4/3_ON78 | 192.76 | 34.4 | 158.36 | 1920 | 35.2 | 33.4 | 941/958 | 8 | 77 | 23909 |  |

64K: n ON=2 OFF=2 | makespan raw ON 165.13 OFF 170.00 -> +2.95% | pause-subtracted ON 137.98 OFF 135.60 -> -1.73% | tok/decode-s all ON 35.4 OFF 39.0 -> -9.1% | paired non-void turns 4: ON 27.4 OFF 30.7 -> -10.6%

64K (full-token runs only): n ON=1 OFF=2 | makespan raw ON 171.59 OFF 170.00 -> -0.93% | pause-subtracted ON 141.39 OFF 135.60 -> -4.09% | tok/decode-s all ON 44.1 OFF 39.0 -> +13.2% | paired non-void turns 4: ON 46.2 OFF 30.7 -> +50.6%

78K: n ON=2 OFF=2 | makespan raw ON 193.06 OFF 189.21 -> -2.00% | pause-subtracted ON 158.21 OFF 154.41 -> -2.41% | tok/decode-s all ON 34.0 OFF 33.1 -> +2.7% | paired non-void turns 4: ON 32.5 OFF 25.0 -> +29.9%

78K (full-token runs only): n ON=2 OFF=2 | makespan raw ON 193.06 OFF 189.21 -> -2.00% | pause-subtracted ON 158.21 OFF 154.41 -> -2.41% | tok/decode-s all ON 34.0 OFF 33.1 -> +2.7% | paired non-void turns 4: ON 32.5 OFF 25.0 -> +29.9%


## Gate verdicts
| gate | result |
|---|---|
| greedy tokens identical to OFF | NOT JUDGED (two-slot runs not deterministic: OFF-vs-OFF turn token_sha 8/12; ON78 run_sha identical across its 2 runs, OFF78 not) |
| decode at 64K >= +5% | INCONCLUSIVE: all-turn -9.1%, full-token runs +13.2% (n=1), same-arm variance about 2x |
| makespan at deep ctx >= +1% | FAIL: 78K pause-subtracted -2.41% (raw -2.00%), n=2 per arm, ABBA |
| makespan at 64K | FAIL: pause-subtracted -1.73% (raw +2.95% is a pause and short-run artifact) |
| VRAM 2x96K at ub1024 | FAIL: ON96 OOM at startup prefill (peak 24410 MiB). Max fit 2x79872 (ON peak 23909 MiB, matches 23.9 GB prediction) |

**Verdict: REJECT.** The kill condition is met (makespan <=0 at 64K and 78K, MTP buffers force ctx <96K). No adopt flags.

## Notes and risks
- Run order: abba3 ABAB (OFF64, ON64, OFF64, ON64, then ON96 OOM). abba4 ABBA (OFF78, ON78, ON78, OFF78). Stop at 08:15 before OFF64/ON64 repeats.
- Possible early-EOS bug, OPEN: abba3/2_ON64 stopped on EOS in all agent-1 turns (down to 1-2 tokens) plus agent-0 turn 5, 1412/1920 tokens. Not reproduced in 3 other ON runs or any OFF run. Not diagnosed.
- Pacing 77/72 C, max 15 s, both arms: pause in timed phase 24-36 s per run, subtracted. tmax 77 in all valid runs.
- MTP is wired for -np 2 (server-context.cpp draft loop over all slots; one batched MTP draft). No patch.
- No commits (branch claude/speed-c1-20261122 @7ac90aa), no build. Stock combined-r3-card1 libs; per-run sha256 in <run>/runtime-library-hashes.txt.
