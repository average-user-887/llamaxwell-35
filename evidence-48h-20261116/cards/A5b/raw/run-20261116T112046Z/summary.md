| policy | n | makespan mean s | sd s | vs policy 0 | slot0 end mean s | slot1 end mean s | peak VRAM max MiB | worst gap max s | token-identical to p0 ref |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 160.26 | 1.82 | +0.00% | 154.40 | 160.26 | 23867 | 5.67 | 4/6 |
| 4 | 6 | 160.89 | 1.95 | +0.40% | 154.99 | 160.89 | 23867 | 11.33 | 0/6 |

ACCEPT policy 4 peak VRAM 23867 <= p0 peak 23867 + 100 MiB: PASS
ACCEPT policy 4 worst visible gap 11.33 s <= 10 s: FAIL
(p0 worst gap max 5.67 s)

Per session (tag b<block>_p<position>):

| session | policy | makespan s | slot0 end s | slot1 end s | peak MiB | worst gap s | identity vs p0 ref | engagement |
|---|---|---|---|---|---|---|---|---|
| b0_p0_p0 | 0 | 158.64 | 152.86 | 158.64 | 23867 | 5.60 | identical | (none; policy 0) |
| b0_p1_p4 | 4 | 161.05 | 155.10 | 161.05 | 23867 | 11.27 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1929 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17199.8 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=759 decode_steps_2row=1154 |
| b1_p0_p4 | 4 | 161.63 | 155.67 | 161.63 | 23867 | 11.25 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1930 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17217.4 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=761 decode_steps_2row=1153 |
| b1_p1_p0 | 0 | 163.25 | 157.11 | 163.25 | 23867 | 5.67 | slot0: FIRST DIVERGENCE turn 2 pos 5 (differs); slot1: FIRST DIVERGENCE turn 2 pos 73 (differs) | (none; policy 0) |
| b2_p0_p4 | 4 | 164.37 | 158.25 | 164.37 | 23867 | 11.33 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1929 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17335.6 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=759 decode_steps_2row=1154 |
| b2_p1_p0 | 0 | 161.74 | 155.78 | 161.74 | 23867 | 5.60 | slot0: identical (4 turns, 1536 tokens); slot1: FIRST DIVERGENCE turn 2 pos 158 (differs) | (none; policy 0) |
| b3_p0_p0 | 0 | 159.17 | 153.41 | 159.17 | 23867 | 5.60 | identical | (none; policy 0) |
| b3_p1_p4 | 4 | 159.59 | 153.76 | 159.59 | 23867 | 11.27 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1928 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17182.9 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=757 decode_steps_2row=1155 |
| b4_p0_p0 | 0 | 159.43 | 153.67 | 159.43 | 23867 | 5.62 | identical | (none; policy 0) |
| b4_p1_p4 | 4 | 159.48 | 153.71 | 159.48 | 23867 | 11.27 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1929 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17200.2 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=759 decode_steps_2row=1154 |
| b5_p0_p4 | 4 | 159.25 | 153.46 | 159.25 | 23867 | 11.27 | slot0: FIRST DIVERGENCE turn 1 pos 272 (differs); slot1: FIRST DIVERGENCE turn 0 pos 46 (differs) | policy=4 chunk=512 maxdefer=64 batches_seen=1929 batches_deferred=0 defer_events=0 batches_capped=0 rows_removed_by_cap=0 HOLD hold_ms=150 hold_b=4 reach=4096 holds=5 hold_ms_total=17155.9 steps_merged=5 empty_batch_fallbacks=0 decode_steps_1row=759 decode_steps_2row=1154 |
| b5_p1_p0 | 0 | 159.34 | 153.57 | 159.34 | 23867 | 5.60 | identical | (none; policy 0) |

Reference = b0_p0_p0. makespan = last request end - t_run0. Peak = max of 5 s whole-GPU samples inside the session window. gap = max inter-event gap within one streamed request.
