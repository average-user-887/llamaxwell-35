# Native MTP resume, first measurements

The first 80K serial process exceeded its startup readiness window and completed owned cleanup. It produced no measured output. The second completed both requests and cleanup on the same pinned c973 CUDA runtime. Cached decode reached **76.7431 tok/s** with **40.16078 s** whole-request time. The cold request took **262.87950 s**. It produced 192 outputs cold and 3,072 warm, reusing 79,996 warm prompt tokens and evaluating four. The prompt was a fixed throughput fixture with no independent semantic scoring key.

Both attempts retain every typed trace-v2 event. Unknown per-cycle costs remain null. The first trace records missing request outcomes. The second includes serial cycles, state markers, stream checkpoints and output boundaries. Private token payloads, text, their digests and local identifiers were removed.

## Three-row MoE operator

Two synthetic fixtures exercised independent and partially overlapping routes with eight experts per row. The actual 48-block candidate ran with two resident CTAs per SM on 24 SMs. All 12,288 cohort outputs were finite and bit-identical to separate legacy V3 rows. The receipt includes all measured intervals and launch counts, actual occupancy and the finite generic/V3 numerical difference.

| Fixture | Generic minus cohorts, µs/window | Sequential V3 minus cohorts, µs/window |
| --- | ---: | ---: |
| Independent | 139.854, 151.330 | 64.897, 80.442 |
| Partial overlap | 135.387, 145.823 | 74.118, 87.840 |

Both reversed pairs passed 116.5 µs and 30 µs thresholds respectively. Each timing interval used 96 synchronized evaluations, with 96 physical launches in each candidate interval and zero in the controls. The default-off candidate preserves 16 existing device functions and 42,664 instruction and control words. The unmodified relink reproduced c973 byte for byte. Across 40 representative layers the projected saving is 5.42–6.05 ms per verifier round. These are standalone operator measurements. Whole-model throughput, engagement and output quality remain open.

The requested Fable advisory was rejected because the service credit balance was too low. No advice was returned. Historical Claude and Fable figures remain separate from these fresh receipts and ratios.

See [reproduction](REPRODUCE.md) and [manifest](MANIFEST.md).
