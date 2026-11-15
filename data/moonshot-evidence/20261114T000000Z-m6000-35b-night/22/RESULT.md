# Card 22 RESULT: phase split of N3 replay (card 16 runs)

Per slot, mean over runs. pre = sum of (request send -> first token), dec = sum of (first token -> end), idle = makespan - pre - dec (start offset, 1 s think gaps, tail). stall = other slot's token-gap overlap with this slot's prefill window (gaps > 3x normal decode step).

| arm | n | slot | appends | prefill s | decode s | idle s | sum s | makespan s | longest stall of other slot s | stalled-overlap sum s |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 5 | 0 | 4 | 65.0 | 87.6 | 9.0 | 161.6 | 161.6 | 5.57 | 11.4 |
| S | 5 | 1 | 4 | 77.9 | 55.6 | 28.0 | 161.6 | 161.6 | 5.59 | 43.5 |
| M | 5 | 0 | 4 | 64.0 | 86.8 | 8.8 | 159.6 | 159.6 | 5.58 | 16.8 |
| M | 5 | 1 | 4 | 71.6 | 60.0 | 28.0 | 159.6 | 159.6 | 5.61 | 43.7 |

| arm | max token gap s (any slot) | union of prefill windows s | prefill-window overlap between slots s | normal decode step ms (slot0/slot1) |
|---|---|---|---|---|
| S | 5.59 | 108.8 | 34.1 | 31.9 / 31.9 |
| M | 5.61 | 108.0 | 27.7 | 31.2 / 31.3 |

Max gap example (M_2): 5.60 s in request a0_t3_4K slot 0 at t=141.6-147.2 s; overlapping other-slot prefill: [('a1_t3_4K', 1, 125.0, 147.4)].

## Conclusion
1. Makespan 160 s: slot0 prefill/decode/idle 64/87/9 s, slot1 72/60/28 s; slot1 idle is mostly its 25 s start offset, slot0 idle is 1 s think gaps plus tail.
2. The max gap (5.61 s) is one slot's decode stream frozen while the other slot prefills a 4K append; longest stalls per slot 5.6/5.6 s.
3. Stalled decode overlapping the other slot's prefill sums to 16.8 s (slot0) and 43.7 s (slot1); both slots prefill concurrently for 28 s.
4. GPU is not idle in these stalls (prefill compute-bound); recoverable only via mixed prefill+decode batches or smaller chunks. Upper bound if stalled decode were free: ~44 s on the critical slot; realistic fraction well under half.
5. Best case from scheduling alone is an estimate, roughly 3-8 s (2-5%), unmeasured; the dominant cost is prefill throughput (~200-270 tok/s) and decode step time, not idle.
