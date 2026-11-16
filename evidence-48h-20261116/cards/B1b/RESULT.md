# B1b (ChatML-wrapped fixture): ABORTED after offA by the void guard
| arm | status | void turns | makespan s | note |
|---|---|---|---|---|
| offA | ran (04:06-04:18 CET 16 Nov) | 12/12 (all loop >50% repeated 4-grams, 800 tokens, stop=limit) | see run.json | peak VRAM 22982 MiB |
| offA2, mtp2, mtp3, mtp1, ngram-mod, ngram-map-k | not run (abort, exit 5) | n/a | n/a | |
| ngram-mod+MTP2 | dropped | n/a | n/a | OOM at ub1024/84224 (from B1) |

No accept criteria evaluated: no spec arm and no floor compare ran.

Method: b1b.sh = B1 script with 7 arms, void abort after offA, floor-mode n3r_compare + mtp2-vs-mtp1 at the end; flock -w 43200; unload via POST qwen36-35b-moe only; containers nb1-*, none left.
Caveat: the ChatML wrap (chat_wrapped=True) did not stop the temp-0 loops: both slots hit the 800-token limit on every turn, with repeated token shas across turns (e.g. 2501987f..., 08ab3086...). Suggested next: --extra-body '{"repeat_penalty":1.05}' (README B0b) or a different prompt/stop strategy; do not rerun the matrix until offA shows void_turns <= half.
Files: <W>/B1b/run-20261116T012631Z/steps.log, <W>/B1b/run-20261116T012631Z/offA/run/{run.json,turns.md}, <W>/B1b/b1b.sh, <W>/B1b/summarize.py
