# B1c: ABORTED after S_offA (void guard)
| arm | status | void turns | note |
|---|---|---|---|
| S_offA (temp 0.7 seed 1234 n400) | ran 04:33-04:43 CET 16 Nov | 10/12 | peak VRAM 22882 MiB |
| E_offA, S_mtp1, S_mtp2, S_ngmod, E_offA2, E_mtp1, E_mtp2 | not run (exit 5) | n/a | |

Void breakdown (S_offA): loops (>50% repeated 4-grams) on 5 turns incl. 3 that hit the 400 limit; short (<64 tokens, EOS) on 5 turns (7, 9, 10, 35, 35 tokens); only 2 turns (262, 261 tokens, prose) were clean. No accept criteria evaluated.
Files: <W>/B1c/run-20261116T031936Z/steps.log, <W>/B1c/run-20261116T031936Z/S_offA/run/{run.json,turns.md}, <W>/B1c/b1c.sh, <W>/B1c/summarize.py
