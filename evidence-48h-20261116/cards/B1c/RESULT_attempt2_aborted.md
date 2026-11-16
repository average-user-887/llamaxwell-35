# B1c attempt 2 (B0c fixture): ABORTED after S_offA (void guard)
| arm | status | void turns |
|---|---|---|
| S_offA (temp 0.7 seed 1234 n400) | ran 05:53-06:04 CET 16 Nov, peak VRAM 22882 MiB | 8/12 |
| E_offA, S_mtp1, S_mtp2, S_ngmod, E_offA2, E_mtp1, E_mtp2 | not run (exit 5) | n/a |

Breakdown: loops (>50% repeated 4-grams) on 6 turns (agent0 turns 0,3; agent1 turns 0,1,3,4), all 400-token limit; 2 short EOS turns (agent1 turn 2 and 5: 9 and 8 tokens). Clean: 4 turns (agent0 turns 1,2,4,5, ~400 tokens, no EOS). Improved vs attempt 1 (10/12) but still >50% void. No accept criteria evaluated.
Files: <W>/B1c/run-20261116T034617Z/steps.log, <W>/B1c/run-20261116T034617Z/S_offA/run/{run.json,turns.md}; attempt 1 in RESULT_attempt1_aborted.md
