# Split draft re-evaluation on the Sparse R2 MTP stack

On a fixed 80,000-token prompt, switching `LLAMA_MTP_MERGE_REEVAL` from `1` to `0` raised warm server decode from **77.7706 to 82.6304 tok/s (+6.25%)** and reduced warm request wall from **2.71705 to 2.57900 s (−5.08%)**. Each mode reused 79,996 prompt tokens and produced the same 192 output IDs. Cold wall rose by **1.5771 s** despite higher cold decode throughput.

| 80K result | Merged | Split |
|---|---:|---:|
| Cold prefill (s) | 265.6840 | 267.3447 |
| Cold request wall (s) | 268.1223 | 269.6994 |
| Cold server decode (tok/s) | 79.0614 | 82.1493 |
| Cold draft accepts | 90/100 | 89/101 |
| Warm request wall (s) | 2.71705 | 2.57900 |
| Warm server decode (tok/s) | 77.7706 | 82.6304 |
| Warm server decode time per proposed draft token (ms) | 23.8441 | 22.8861 |
| Warm draft accepts | 87/103 | 89/101 |

The 80K run used one cold request and one same-slot warm replay in each mode. The split arm's cold prefill took 1.6607 s longer. Exact output matching covers the merged and split cold requests and the merged and split warm requests. The fixture has no semantic score. Server decode uses `predicted_per_second`. The per-proposal metric divides total server decode milliseconds by the number of draft proposals. [Allowlisted measurements](evidence.json)

## Graph checks

The separate 2K phase trace recorded 174 merged MTP q3 calls. The split trace recorded no MTP q3 calls, 206 q1 calls, and 205 q2 calls. Both arms recorded 205 target q2 calls. Both 2K comparisons matched all 192 output IDs and retained 88/102 draft accepts in each condition. Phase counts provide diagnostic graph evidence. The 80K timing runs kept the ordinary host counters and left phase and node timers disabled.

Both 80K arms offloaded 42/42 layers, used the same hashed server and libraries, and emitted live 1,024-row cache-only and long-context Sparse R2 q1/q2 markers. These markers record path execution. Total sparse dispatch coverage remains unmeasured.

## Reproduce the configuration

Use the published Sparse R2 plus cache-only stack with the model and runtime settings in the receipt. The only environment change between the two arms was `LLAMA_MTP_MERGE_REEVAL=1` to `LLAMA_MTP_MERGE_REEVAL=0`. Keep `LLAMA_MTP_CACHE_ONLY=1`, both Sparse R2 flags, the single slot, 84,224-token context, batch size 2,048 and microbatch size 1,024. Keep MTP1 with F16 draft K/V. The existing stack source supports both settings. [Published source recipe](https://github.com/average-user-887/llamaxwell-35/blob/5ec57de6c5b988b83f3cfe7fd5ac9e476b3e8880/stacked-mtp-20261118/README.md)

```bash
# Use the normal stack arguments and shared environment for both arms.
# Set 1 for merged and 0 for split.
export LLAMA_MTP_MERGE_REEVAL=0
export LLAMA_MTP_CACHE_ONLY=1
```
