# Same-build native width controls and four-row attention

The fresh, fixed c973 runtime used one 80K slot, every expert, the same sparse attention setting, and the same deterministic sampler. Only the native draft configuration changed between the four uninstrumented whole-model arms. Each completed both requests and cleaned up. The first serial attempt failed at readiness and was already published with the successful repeat. [Serial source lineage and both attempts](../native-mtp-resume-first-20261118/README.md)

| Mode | Warm tok/s | Warm wall s | Cold wall s | Warm accepted / offered | Warm native cycles | Attributed emissions / native cycle |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 76.7431 | 40.1608 | 262.8795 | — | — | — |
| Depth 1 | 83.3494 | 37.1346 | 269.1401 | 1449 / 1621 | 1621 | 1.894 |
| Depth 2 | 71.9299 | 42.9769 | 268.4224 | 1878 / 2386 | 1193 | 2.574 |
| Depth 3 | 60.3770 | 51.1489 | 269.6083 | 2183 / 2661 | 887 | 3.461 |

All warm requests emitted 3,072 IDs. Traces retain separate boundary and unattributed emissions. Full decode divided by native cycles was 22.74, 35.80 and 57.36 ms at depths one, two and three. Full checkpoint-restore counts were zero. The ordinary fixture is an unscored forced continuation. The first warm differences versus serial were at tokens 27, 61 and 19. The receipts record nonempty and repeated lines without reproducing the text.

The separate four-row attention test used the new fd436e CUDA build at 81,920 keys, without a full-model request. Measured Q1/Q2/Q3 operator times were 186.80, 248.34 and about 363.93 µs. Stock Q4 was about 1,782.36 µs. Both sparse Q4 layouts passed timing and numerical gates. NU2 was selected at 413.635 µs median. Numerical validation checks all finite outputs against two independently dispatched two-row calls, exact row/KV plans and the prior selector policy. This compares the attention component and leaves whole-model quality and throughput to separately matched runs.

[Native release commands](REPRODUCE.md) · [Four-row operator commands](Q4-OPERATOR.md) · [Exact Q4 patch](q4-attention.patch)
