# Card 27: does the N3 replay re-evaluate already-held prompt tokens on each append?

| slot | appended tokens | evaluated tokens | excess | est. seconds at 590 t/s |
|---|---|---|---|---|
| 0 (per run, all 18 runs identical) | 10246 | 10246 | 0 | 0.0 |
| 1 (per run, all 18 runs identical) | 10246 | 10246 | 0 | 0.0 |
| total over 18 runs x 2 slots | 368856 | 368856 | 0 (0.00%) | 0.0 |

Per-turn detail (identical in every run): turns append 1024, 4098, 1026, 4098 tokens; the server's
first "cached n_tokens" for each task equals the previous task's final n_tokens, so the reused prefix is exact.
Run time per run is about 120 s of prompt eval per two slots; the excess is 0 s, i.e. 0% (threshold 2%).

**VERDICT: DEAD** (no prompt re-evaluation, no cache miss, no checkpoint re-restore during replay turns)

## Scope
18 valid batch8 runs (run.json ok=true): card 24 M_2/4/6/8/10 and S_1/3/5/7/9, card 25 A_1/6/8, A2_2/5/7, Q_3/4
(arms: prod config, 2 slots, 84000 start, 4 turns, cache-ram 0, ctx-checkpoints 16).

## Log lines used (server.log)
- `slot operator(): id N | task T | new prompt, ... task.n_tokens = X` : prompt length
- `... | cached n_tokens = Y, memory_seq_rm [Y, end)` (first per task): reused prefix; evaluated = X - Y
- `slot print_timing: ... prompt eval time = ... ms / K tokens`: tokens actually evaluated (K equals X - Y in every task)
- `slot release: ... stop processing: n_tokens = E`: end of previous turn on that slot; appended = X - E
  (first turn: appended = X - restored 84000)
- `slot load_slot_ch: ... restored 2 context checkpoint(s) from ...bin`: only the initial slot restore (36 total = 18 runs x 2 slots)
- Searched for `forcing full`, `re-process`, `invalidat`, `erased`, `cache miss`, `memory_seq_rm [0`: only hits are the 8-token warm-up task 0 (`cached n_tokens = 0`), 18 times.

Script: reuse_check.py (this directory). Append sizes match n3_replay.py (1K=1024 / 4K=4096 tool chunks plus a newline token and sampled token).
