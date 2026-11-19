# Native 80K compact-readback shadow refusal

The initial frozen native depth-one MTP shadow ran to a terminal refusal. The mapped CUDA, core and server builds matched the candidate profile, with the pinned original executable and common sampler library. Down4 issued 4,080 cold and 65,840 warm calls. The compact request, launch, transfer and row comparison counters were all zero in both phases despite the mapped child environment requesting shadow mode.

| Observation | Cold | Warm |
| --- | ---: | ---: |
| Requested prompt tokens | 80,000 | 80,000 |
| Cached prompt tokens | 0 | 79,996 |
| Requested generation tokens | 192 | 3,072 |
| Token IDs observed by SSE client | 192 | 3,071 |
| Server predicted tokens | 192 | 3,072 |
| Candidate draft offers / accepts | 102 / 88 | 1,646 / 1,424 |
| Archived scorer control offers / accepts | 101 / 89 | 1,642 / 1,429 |
| Exact token sequence and text match | Yes | No |
| First token mismatch, one-based | — | 1,603 |
| Client time to first token, seconds | 266.981571 | 0.275277 |
| Client request wall time, seconds | 269.109716 | 31.533846 |

Both requests reached the server's generation limit with truncation false. The warm token-bearing SSE checkpoint at 3,072 was absent. The validator recorded the pinned-protocol and SSE checkpoint refusal. Runner exit was 1 and cleanup exit was 0. The bounded run was 374 seconds against a 430-second cap. The earlier mismatch at 1,603 is a separate observation from the final SSE count discrepancy.

The initial common parser and server guard explain why the compact path stayed idle. In the archived build's `common.h`, the vector defaults to `[NONE]`. The `--spec-type` handler in `arg.cpp` appends `DRAFT_MTP`, and subsequent automatic inference only applies to the original single-element default. The initial candidate guard in `server-context.cpp.patch` requires exactly one speculative type and that type must be `DRAFT_MTP`. The actual child argument therefore leaves `[NONE, DRAFT_MTP]`, which rejects the compact path before the request counter. The cause of the output drift and its reproduction with a down4-only build remain unresolved at this packet's boundary.

This shadow attempt supplies behavioral and refusal evidence. Model throughput and compact exactness remain unqualified.
