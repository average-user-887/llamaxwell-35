# Depth-two native MTP cycle timing at 80K

The instrumented run completed on the Quadro M6000 and matched all 3,264 output IDs and the complete text of its uninstrumented same-depth reference. The cold request generated 192 IDs. The warm request generated 3,072 after reusing 79,996 prompt tokens and evaluating four. Runner and cleanup both exited successfully.

The warm request recorded 1,193 verifier attempts. The draft proposed 2,386 IDs, of which 1,878 were accepted. It emitted 3,071 IDs in those attempts and one initial serial ID. Across cold and warm requests the full direct timing trace contains 1,267 records.

| Warm stage | Mean per verified attempt |
| --- | ---: |
| Draft | 4.524 ms |
| Target decode API | 11.952 ms |
| Existing target output wait | 18.160 ms |
| Draft cache process | 0.200 ms |
| Sample and accept | 0.471 ms |
| Direct target trim | 0.089 ms |
| Direct draft trim | 0.039 ms |
| Other time in the attempt | 0.370 ms |
| Complete measured attempt | 35.805 ms |
| Inter-attempt gap recorded separately | 0.009 ms |

The direct target trim averaged 87.777 µs for 857 all-accepted attempts and 92.729 µs for 336 native rejections. Both are host API measurements. There were zero checkpoint saves, loads, restores or replay attempts in this run. Deferred physical GPU rollback time is unknown and stays null in the data.

The timer adds host measurements to existing call boundaries. The target output wait uses the synchronization already present in the server. GPU phase profiling was off. Use the uninstrumented receipt as the throughput control. See [the timing and parity summary](cycle-timing.json), [all underlying direct timing rows](cycle-timing.jsonl), [the typed verifier trace](trace-v2.jsonl), [runtime receipt](depth2.receipt.json), and [portable reproduction](REPRODUCE.md).
