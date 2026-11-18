# Q8A-enabled serial control attempt

The same-build Q8A-enabled no-MTP control completed 192 cold and 3,072 warm outputs. It reported 73.6983 and 76.4693 tok/s, with cold and warm request times of 263.1234 and 40.3026 seconds. Startup verified the mapped CUDA binary and related runtime, one slot, eight experts, no MTP context and 42/42 GPU offload. HTTP reported 79,996 warm cache tokens and four evaluated tokens. The retained late-warm server log showed long sparse Q1 dispatch.

The original run exited 1 during verification and cleanup exited 0. The 4 MiB single-file Docker log rotated between the cold snapshot and final capture, leaving a gap at the warm request start. The request-local checkpoint restoration marker required by the strict verifier was absent. The independent check retained both output timings and the original validation refusal. The strict native-to-serial comparison remains open.

The [archived native Q8A-on reference](native-on.reference.json) measured 86.2633 tok/s and 35.8766 seconds on its warm depth-one request. The available [native trace](native-on.trace-v2.jsonl) has 70 complete cumulative host snapshots. The [serial trace](serial.trace-v2.jsonl) records the lost log continuity and the independent stream counts and checkpoints. The native and serial outputs first differed at ID 170 in each request. These timings are scoped to one forced, unscored continuation. Separate scored exercises remain in the [original native package](../gdn-q8-warm80k-20261118).

| Warm output | Digit-normalized duplicate lines | Duplicate lines in final quarter | Trailing comment-only lines | Repeated 24-word windows |
| --- | ---: | ---: | ---: | ---: |
| Serial control attempt | 73/283 | 16/71 | 0 | 46/1,318 |
| Archived native Q8A-on | 127/304 | 39/76 | 0 | 63/1,336 |

The same-input warm native output showed more repetition under these counts. See [the full descriptive counts](repetition.json) and [independently verified partial serial observations](partial-observations.json).
