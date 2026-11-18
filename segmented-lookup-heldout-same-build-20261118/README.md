# Same-build serial control for held-out 32K segmented lookup

On the held-out 32K code revision, segmented lookup ran **2.221× faster in warm decode** than serial on the same build. It finished the warm request in **6.806 seconds versus 14.989 seconds**, a **2.202×** whole-request speed ratio. All six measured responses returned the same **1,087 tokens through EOS** and matched the independently fixed complete source text and Python AST.

| Warm condition | Decode rate | Decode time | Request wall |
| --- | ---: | ---: | ---: |
| Serial, separate process | 73.01207 tok/s | 14.88795 s | 14.98939 s |
| Single source range, median of two same-slot controls | 139.33060 tok/s | 7.80162 s | 7.90959 s |
| Segmented, one same-slot request | 162.13872 tok/s | 6.70414 s | 6.80575 s |

The lookup process ran a single-range cold prime, then warm single-range, segmented, and single-range requests. The serial process ran one cold request and one warm request. The cold serial request took 91.782 seconds and the lookup process's cold single-range prime took 85.108 seconds. No cold segmented request was measured. All warm requests reused **32,764 tokens** and evaluated **four**. The live runtime checks verified the same mapped build and profile, and both runs completed cleanly.

The segmented request accepted all **1,035** draft tokens and logged no checkpoint restore. Each single-range control accepted **1,040 of 1,174** and restored one checkpoint before q123 replay. Serial logged zero drafts and verifier calls. This is one held-out revision on a 32K prompt, with one warm serial and one warm segmented request. A separate 80K revision previously cleared 2× under its own same-build comparison. [Independent 80K result](https://github.com/average-user-887/llamaxwell-35/blob/3ded5dda2309112f1fe5fb248caa7a50f69ef3cd/segmented-lookup-80k-20261118/README.md)

## Reuse the published build

Follow the existing source reconstruction. The lookup process used `ngram-simple` with key length 16, maximum draft 255 and one minimum hit. The serial process used `--spec-type none` and omitted both lookup fields. Other server arguments and numeric environment settings matched. Both profiles used one slot, full F16 KV, no context shift, greedy seed 7, prompt caching and EOS stopping. The [allowlisted receipt](evidence.json) records the measured configuration and per-request metrics. [Published segmented lookup implementation](https://github.com/average-user-887/llamaxwell-35/blob/2c32420080d5d151d34191743041bc1e7a1379cb/segmented-lookup-32k-20261118/README.md)

The held-out ranges came from the original committed source, excluding the tokens overlapping the requested edit and the standalone terminal LF. The prompt and rendered template matched across runs. The expected full source and the previous serial output were used after generation to verify the result. [Held-out input-only derivation and same-slot evidence](https://github.com/average-user-887/llamaxwell-35/blob/a9a5117e4d7a0a6a9ce4a66fb4f5440d6fe1e6b6/segmented-lookup-heldout-32k-20261118/README.md)
