skipped (not ok): <W>/D1b/run-20261116T134833Z/runs/b0_p1_gr/run.json  GEMM_NOT_ENGAGED
| arm | n | makespan mean s | sd s | peak VRAM max MiB | token-identical to prod ref | compute_calls (mean) | graph_launches (mean) | captures (mean) |
|---|---|---|---|---|---|---|---|---|
| prod | 1 | 159.53 | nan | 23867 | 1/1 | 1951 | 0 | 0 |

Per session:

| session | arm | makespan s | peak MiB | calls | arch_off | captures | launches | per-slot identity vs prod ref |
|---|---|---|---|---|---|---|---|---|
| b0_p0_prod | prod | 159.53 | 23867 | 1951 | 1951 | 0 | 0 | slot0 identical (4 turns, 1536 tokens); slot1 identical (4 turns, 1536 tokens) |

makespan = last request end - t_run0; peak = max whole-GPU 5 s sample in the session window. Counters from the last d1b_graphs line of each server.log (periodic every 5000 launches + exit).
