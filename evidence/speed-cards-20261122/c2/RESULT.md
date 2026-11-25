# C2 append scheduling (LLAMA_APPEND_SCHED=1): REJECT

| run | arm | makespan raw s | net of pause s | pause in run s | p95 gap a0/a1 s | max gap a0/a1 s | void | tmax C | sha |
|---|---|---|---|---|---|---|---|---|---|
| 1 | OFF84 | 199.2 | 165.04 | 34.2 | 0.057/0.057 | 0.60/4.99 | 8 | 77 | 04e05e5605 |
| 2 | ON84 | 199.0 | 163.67 | 35.3 | 0.057/0.057 | 5.59/6.97 | 8 | 77 | 0bc4ee4495 |
| 3 | ON84 | 199.8 | 164.55 | 35.3 | 0.057/0.057 | 6.71/6.95 | 8 | 77 | 0bc4ee4495 |
| 4 | OFF84 | 202.4 | 162.0 | 40.4 | 0.057/0.057 | 1.07/1.07 | 8 | 77 | 9b770d9c7a |

84K net of pause: OFF 163.5 s | ON 164.1 s | gain -0.36%

84K: OFF n=2 mean 200.8 s | ON n=2 mean 199.4 s | gain +0.70% | p95 stall a0 0.057->0.057 (-0.2%) a1 0.057->0.057 (-0.1%)

Valid runs only: n=2 per arm (OFF84, ON84), ABBA order, all at tmax 77 C, peak VRAM 24113 MiB, void=8 in every run on both arms. Excluded: hot80/ (5 runs at 80 C) and oom96/ (3 runs, 2x96K OOM).
Starve-overrides = 0 in both ON runs (16 owner switches per run).

| gate | result |
|---|---|
| 84K makespan >= +1% | FAIL: +0.70% raw, -0.36% with pauses removed |
| 2x64K not worse | NOT RUN (stopped at 08:15) |
| KLD / MV2 logprob proxy | NOT RUN |
| p95 decode stall <= +5% | PASS (0.057 s on both arms) |
| max stall (not a gate) | worse on ON: 5.6-7.0 s vs 0.6-5.0 s |

## Verdict
REJECT. The gain is under the 1% kill line and turns negative once thermal pauses are removed. Interleaving already exists through --mixed-batch-size, and shortest-first ownership adds nothing measurable on the 84K append fixture. Do not promote.

## Risks and limits
- Only n=2 per arm, and pause time (34-40 s per run) is large next to the effect.
- The p95 stall includes pause time, the same on both arms.
- The remaining-token estimate for a started slot can be low after a checkpoint rollback. The 8-batch anti-starvation bound never fired.

## Build
- Branch claude/speed-c2-20261122 (local only). Commits 997feee, 9bf03e1, c6e3cc2.
- ON libllama-server-impl.so sha256 f9aaae14cb24cc62454d14698dae06de382c3c2b31080b137fb67df2ac99bfab. The control build is card1 byte-exact.
- Full hashes and the build command are in RECEIPT.txt.
