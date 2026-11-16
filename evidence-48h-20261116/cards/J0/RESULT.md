# J0 ledger analysis (read-only), run-20261116T071516Z

| Metric | 1-row (n_tokens=1) | 2-row (n_tokens=2) |
|---|---|---|
| Decode steps per replay (L_1 / L_2 / L_3) | 751 / 751 / 751 | 1149 / 1149 / 1149 |
| Share of decode steps (1900 per replay) | 39.5% solo | 60.5% |
| Mean period_ms per replay (L_1 / L_2 / L_3) | 18.29 / 18.31 / 18.23 | 31.57 / 31.61 / 31.58 |
| Mean wall_ms per replay (L_1 / L_2 / L_3) | 17.88 / 17.89 / 17.84 | 30.86 / 30.88 / 30.89 |
| Total steps, 3 replays | 2253 | 3447 |
| Sum period_ms, 3 replays | 41,180.9 ms (27.4%) | 108,877.2 ms (72.6%) |
| Sum wall_ms, 3 replays | 40,257.2 ms (27.5%) | 106,427.7 ms (72.5%) |

Idle windows (gap_ms >= 2000), L0, all replays: 5 windows, sum 31,129 ms.

| Replay | gap_ms | Preceding row | n_prompt / n_gen / n_tokens |
|---|---|---|---|
| L_1 | 9243 | decode | 0 / 1 / 1 |
| L_2 | 9123 | decode | 0 / 1 / 1 |
| L_2 | 2004 | prefill | 2048 / 0 / 2048 |
| L_3 | 8754 | decode | 0 / 1 / 1 |
| L_3 | 2005 | prefill | 2048 / 0 / 2048 |

Summary: 3 of 5 gaps (27,120 ms) follow a 1-row decode; 2 of 5 (4,009 ms) follow a 2048-token prefill batch.

L1 (.stages, run-20261116T084807Z): 9 http_recv events per replay. Inter-arrival gaps of at least 2 s, per replay: 7 gaps, 9.5 to 36.5 s (L_1: 18.66, 10.98, 14.02, 20.99, 28.54, 15.50, 36.44 s). This is a separate run with no ledger CSV in L1, so these cannot be matched to the L0 idle windows by timestamp.
