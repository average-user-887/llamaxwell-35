# Two-row sparse-union attention on Quadro M6000

On the 81,920-key fixed fixture using the model's 1,024-byte K/V pitch, the F16 sparse-union verifier took **420.84 and 420.88 µs**. The matched control took **175.93 and 176.20 µs**. The candidate added 244.91 and 244.68 µs per full attention call. Both the 70 µs standalone saving and the 50 µs compound screen failed. This ended the experiment at the operator.

| Run order | Width | K/V pitch | Full operator, µs |
| --- | ---: | ---: | ---: |
| Control A | 81,920 | 1,024 | 175.93 |
| Union A | 81,920 | 1,024 | 420.84 |
| Union B | 81,920 | 1,024 | 420.88 |
| Control B | 81,920 | 1,024 | 176.20 |

Across the dense mirrors, both boundary widths and the padded replay, the candidate matched the reference selector scores, metadata and four plans on each fixture. It returned all 8,192 finite values within the declared elementwise and normalized RMS tolerances. The dense 81,920 result had 7,700 changed FP32 words, maximum absolute error 1.304e-8 and normalized RMS error 4.510e-7. Forced invalidation of one row/head replayed 4,096 output words exactly for the affected KV head using the original fallback. The neighboring KV head still ran the candidate. All seven processes completed and the runner and cleanup exited zero.

The merged plans contained 24,568 distinct logical KV-head/token pairs on the canonical fixture. The new kernel staged tile scores, changed warp roles for the value multiplication and doubled the split count. The compiled 16-key loop contains 395 shared loads, 104 shuffles and three CTA barriers. The original four-key loop contains five shared loads, 19 shuffles and zero CTA barriers. These are static counts over different tile sizes and include predicated paths. The grid offered 192 warps across the device versus 384 originally. The timing includes the extra guarded fallback launch and the larger combine. Stage events, hardware traffic and stall counters were unavailable for this run, so the whole-operator difference cannot be allocated among these changes. The throughput field printed by the original harness uses dense FLOPs for its numerator.

Run `python3 -B replay.py` to rederive both timing deltas and verify the numerical markers, plan geometry and completion from the seven process logs. `sha256sum -c SHA256SUMS` authenticates the public packet. The complete chronological logs are in `observed/` with terminal color sequences removed. The published result includes the outcome of the local synthetic-array comparisons. Host-specific mounts and container IDs have been removed. The physical gate, procedure and scope were frozen before the run in [RELEASE.md](RELEASE.md).

The patch applies to the [original stride source](../native-q2-r2-stride-20261118/fattn-occamy-sparse.cuh), and the fixture patch applies to the [original selector harness](../native-q2-r2-selector-20261118/test-backend-ops.cpp). The full revised sources and exact compile/link command templates are included. Run `python3 rehydrate.py /your/container/source /empty/destination` to render the original inputs for the supplied build mount. The original mount reproduced every included frozen file byte for byte. A different mount changes the templated command files and diagnostic paths. Rebuilding and operating the original runner requires its pinned local campaign dependencies. The complete executable identities and compiler resources are in [build.json](build.json). After building, `python3 sass_summary.py control.sass candidate.sass` reproduces [the instruction counts](observed/static-loop.json).
