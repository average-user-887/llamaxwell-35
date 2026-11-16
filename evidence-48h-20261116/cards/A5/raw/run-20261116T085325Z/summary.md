| policy | n | makespan mean s | sd s | vs policy 0 | slot0 end mean s | slot1 end mean s | token-identical to p0 ref |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 160.25 | 0.77 | +0.00% | 154.41 | 160.25 | 1/6 |
| 1 | 6 | 160.53 | 0.93 | +0.17% | 141.03 | 160.53 | 0/6 |
| 2 | 6 | 164.81 | 0.75 | +2.84% | 158.93 | 164.81 | 0/6 |
| 3 | 6 | 162.61 | 0.82 | +1.47% | 136.54 | 162.61 | 0/6 |

Per session (block order = tag b<block>_p<position>):

| session | policy | makespan s | slot0 end s | slot1 end s | identity vs p0 ref | engagement |
|---|---|---|---|---|---|---|
| b0_p0_p0 | 0 | 159.52 | 153.75 | 159.52 | identical | (none; policy 0) |
| b0_p1_p1 | 1 | 159.56 | 139.32 | 159.56 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2319 batches_deferred=586 defer_events=586 batches_capped=0 rows_removed_by_cap=0 |
| b0_p2_p3 | 3 | 161.77 | 135.85 | 161.77 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=715 defer_events=715 batches_capped=11 rows_removed_by_cap=5632 |
| b0_p3_p2 | 2 | 164.08 | 158.28 | 164.08 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1937 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b1_p0_p1 | 1 | 159.64 | 139.39 | 159.64 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2319 batches_deferred=586 defer_events=586 batches_capped=0 rows_removed_by_cap=0 |
| b1_p1_p2 | 2 | 164.16 | 158.36 | 164.16 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1937 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b1_p2_p0 | 0 | 159.16 | 153.39 | 159.16 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | (none; policy 0) |
| b1_p3_p3 | 3 | 161.77 | 135.86 | 161.77 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=714 defer_events=714 batches_capped=11 rows_removed_by_cap=5632 |
| b2_p0_p2 | 2 | 164.29 | 158.50 | 164.29 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1937 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b2_p1_p3 | 3 | 162.07 | 136.08 | 162.07 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=714 defer_events=714 batches_capped=11 rows_removed_by_cap=5632 |
| b2_p2_p1 | 1 | 159.99 | 139.76 | 159.99 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2319 batches_deferred=586 defer_events=586 batches_capped=0 rows_removed_by_cap=0 |
| b2_p3_p0 | 0 | 160.79 | 154.92 | 160.79 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | (none; policy 0) |
| b3_p0_p3 | 3 | 163.50 | 137.29 | 163.50 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=719 defer_events=719 batches_capped=11 rows_removed_by_cap=5632 |
| b3_p1_p0 | 0 | 161.02 | 155.05 | 161.02 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | (none; policy 0) |
| b3_p2_p2 | 2 | 165.86 | 159.76 | 165.86 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1938 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b3_p3_p1 | 1 | 160.86 | 140.39 | 160.86 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2319 batches_deferred=583 defer_events=583 batches_capped=0 rows_removed_by_cap=0 |
| b4_p0_p0 | 0 | 160.17 | 154.30 | 160.17 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | (none; policy 0) |
| b4_p1_p2 | 2 | 165.46 | 159.54 | 165.46 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1936 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b4_p2_p1 | 1 | 161.80 | 146.54 | 161.80 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 73 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2383 batches_deferred=695 defer_events=695 batches_capped=0 rows_removed_by_cap=0 |
| b4_p3_p3 | 3 | 163.41 | 137.24 | 163.41 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=712 defer_events=712 batches_capped=11 rows_removed_by_cap=5632 |
| b5_p0_p3 | 3 | 163.13 | 136.94 | 163.13 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 144 (differs) | policy=3 chunk=512 maxdefer=64 batches_seen=2384 batches_deferred=713 defer_events=713 batches_capped=11 rows_removed_by_cap=5632 |
| b5_p1_p1 | 1 | 161.30 | 140.80 | 161.30 | slot0: FIRST DIVERGENCE turn 1 pos 104 (differs); slot1: FIRST DIVERGENCE turn 2 pos 120 (differs) | policy=1 chunk=512 maxdefer=64 batches_seen=2319 batches_deferred=587 defer_events=587 batches_capped=0 rows_removed_by_cap=0 |
| b5_p2_p2 | 2 | 165.01 | 159.13 | 165.01 | slot0: FIRST DIVERGENCE turn 1 pos 349 (differs); slot1: FIRST DIVERGENCE turn 1 pos 494 (differs) | policy=2 chunk=512 maxdefer=64 batches_seen=1938 batches_deferred=0 defer_events=0 batches_capped=29 rows_removed_by_cap=14848 |
| b5_p3_p0 | 0 | 160.87 | 155.03 | 160.87 | slot0: identical (4 turns, 1536 tokens); slot1: FIRST DIVERGENCE turn 2 pos 24 (differs) | (none; policy 0) |

Reference = b0_p0_p0. makespan = last request end - t_run0 (slot 1 starts 25 s after slot 0). Divergence after the first differing token cascades into later prompts.
