# Native MTP cycle timing at depths one and three

Both 80K runs completed with successful runner and cleanup statuses. Independent receipt comparisons checked all 3,264 IDs and both complete texts at each depth. Each exactly matched its own uninstrumented reference. Both warm requests reused 79,996 cached prompt tokens and evaluated four. See [depth-one parity](depth1.independent-parity.json) and [depth-three parity](depth3.independent-parity.json).

| Warm stage, ms per verified attempt | Depth one | Depth three |
| --- | ---: | ---: |
| Draft | 2.383007 | 6.709183 |
| Target decode call | 1.626138 | 19.532085 |
| Existing target output wait | 17.747336 | 29.509055 |
| Draft cache process | 0.190964 | 0.250363 |
| Sample and accept | 0.392605 | 0.627224 |
| Direct target trim | 0.086389 | 0.089572 |
| Direct draft trim | 0.041468 | 0.040592 |
| Other time in the attempt | 0.345334 | 0.415442 |
| Complete measured attempt | 22.813242 | 57.173515 |
| Separate inter-attempt gap | 0.008314 | 0.010159 |

Depth one recorded 1,621 warm verifier attempts, accepted 1,449 of 1,621 proposals and attributed 3,070 warm outputs to verifier cycles. Depth three recorded 887 attempts, accepted 2,183 of 2,661 and also attributed 3,070. Both warm streams delivered 3,072 IDs. There were 172 and 255 native rejection attempts respectively, with zero checkpoint restores or replays. Direct target trim averaged 92.616 µs over depth-one rejections and 97.961 µs over depth-three rejections. Deferred physical GPU rollback remains null.

The [published depth-two source and reproduction packet](../native-cycle-timing-d2-20261118) records the same instrumentation and build at `b9c6598b8f0b04eeff3e8ae4f34beafc30d2c287`. The host timings retain the original target synchronization and include no GPU phase profiling. The separate [repetition audit](REPETITION.md) describes the forced continuation across the matched Q3 and Q4 model arms.
