# Full-eight-expert native depth one on a single Maxwell GPU

At 80K, full-eight-expert depth-one native MTP delivered **92.00674 cached decode tokens/s**, compared with **76.69489 tokens/s** for same-build serial generation. Both processes loaded the identical Q2-cohort/Q8A runtime and enabled identical flags. Warm throughput rose **19.96463%** and warm client wall fell **6.52559 seconds**. The experiment finished below the 25% improvement gate and the 2× objective. This single comparison includes no counterbalanced repeat or promotion.

| Same-build 80K comparison | Serial, cohorts enabled | Native D1, cohorts enabled |
|---|---:|---:|
| Cold decode, all IDs/s | 74.19708 | 91.93240 |
| Cold request wall, seconds | 264.10558 | 267.55109 |
| Warm decode, all IDs/s | 76.69489 | 92.00674 |
| Warm request wall, seconds | 40.18233 | 33.65674 |
| Startup paired Q2 calls | 78 | 39 |
| Cold paired Q2 calls | 0 | 4,040 |
| Warm paired Q2 calls | 0 | 65,680 |

Both model arms ran to their output limits and completed with successful owned cleanup. The cold request took 3.44551 seconds longer under native MTP despite faster cold token generation. Serial and native text first differed at ID position 170 in both phases. With 3,072 cached output tokens, the serial continuation had 73 duplicate normalized lines out of 283, including 16 of 71 in the final quarter, and 46 repeated 24-word windows out of 1,318. The matched native output had 127/304, 39/76 and 63/1,336 respectively. Serial and native had 170 and 203 Python keyword-led lines, zero trailing comment-only lines and zero malformed characters. These are descriptive fluency signals on one forced continuation. The experiment used no separate task or semantic quality score. The [strict comparison](comparison.json) uses the same all-IDs rate definition for both current arms; its separate historical ratio compares to the earlier HTTP rate, whose numerator omits one token. The [historical recomputation](historical.json) supplies the like-for-like result.

The full-eight-expert Q2 cohort ran on the Qwen/Occamy 35B model with an 80,000-token input and one slot. The candidate produced all 3,264 cold and warm output IDs and both complete texts exactly as the earlier full-eight Q8A native process. This is one forced continuation with a 192-token cold limit and 3,072-token cached limit. Arithmetic and open-ended task quality have no independent score here.

| Native depth one | Earlier Q8A | Q2 cohorts on the same Q8A path |
|---|---:|---:|
| Warm decode, all IDs/s | 86.26331 | 92.00674 |
| Warm wall, seconds | 35.87659 | 33.65674 |
| Warm decode, ms per offered round | 21.68812 | 20.33426 |
| Cold wall, seconds | 268.10631 | 267.55109 |
| Cold accepted / offered | 89 / 101 | 89 / 101 |
| Warm accepted / offered | 1,429 / 1,642 | 1,429 / 1,642 |

The [prior standalone operator](../native-q2-moe-cohorts-20261118) missed its 50 µs per-layer materiality criterion on both fixtures. Observed savings of 29.297–37.550 µs per synthetic graph projected to 1.172–1.502 ms over forty layers. The separate model measurement saved 1.35386 ms per complete warm offered round relative to the archived native baseline. That reference ran in an earlier process. The warm rate increased 6.658% using the same calculation for both, all 3,072 response IDs divided by server-reported decode seconds.

Request-boundary counters show 39 candidate launches and 39 acknowledgments at readiness, followed by exactly 4,040 in the cold request and 65,680 in the warm request. Each physical candidate launch had one second-row acknowledgment; neither legacy row counter advanced. Those deltas exactly equal forty calls per offered draft. The counters cover whole requests and do not identify individual target layers. The Maxwell runtime admitted two resident blocks per SM on 24 SMs. The actual DSO, server executable and loaded model libraries are recorded by digest. The server used the original executable without an interposed profiler.

The archived and candidate native runs emitted identical cumulative acceptance histograms for 65 matched warm snapshots. In their shared 1,600-round window, measured target host time fell by 1.39875 ms per round. Draft, speculative processing and sample/accept times increased by 0.01011, 0.00508 and 0.01080 ms. Cumulative host telemetry includes other target work and lacks exclusive kernel or individual-round duration. See the [matched historical evidence](historical.json) and [native receipt](native-r1.receipt.json).

The initial release reached 42/42 GPU layer offload after 27.201 s, then expired its shared 45-second readiness check. No request started. It exited 2 with successful owned cleanup after 47 seconds. Its record establishes a deadline expiration; the cause of slow startup was unmeasured. The single recovery changed the readiness allowance and associated status text from 45 to 75 seconds in both arms. The overall 430-second lifetime, flags, DSO, exact-output checks, route/counter checks and cleanup stayed identical. The [initial receipt](native-r0.receipt.json) and [two-line recovery](recovery.patch) retain both attempts.

The [portable instructions](REPRODUCE.md) describe source checks, publication verification and the owner-supplied private inputs for reproduction. Published chronology retains available cumulative host events, prompt-cache restore markers, stream clocks, limit stops and all four physical counters at the three request boundaries. Host costs appear only where measured. Raw model output, model-derived ID/stream digests, device state and exported expert IDs are excluded.
