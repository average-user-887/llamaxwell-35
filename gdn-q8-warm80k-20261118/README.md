# Q8_1 GDN on an 80K split-MTP workload

On one matched 80,000-token workload, the Q8_1 Phase A path increased warm decode throughput from **83.476 to 86.263 tok/s (+3.34%)**. Warm request time fell **1.192 seconds**, and server decode time per draft proposal fell **1.015 ms**. The cold request finished 0.190 seconds sooner.

| Measurement | Original GDN T2 | Q8_1 Phase A |
| --- | ---: | ---: |
| Cold request wall, 192 output tokens | 268.29658 s | 268.10631 s |
| Cold decode throughput | 82.54213 tok/s | 86.37358 tok/s |
| Cold drafts accepted / proposed | 89 / 101 | 89 / 101 |
| Warm request wall, 3,072 output tokens | 37.06839 s | 35.87659 s |
| Warm decode throughput | 83.47604 tok/s | 86.26331 tok/s |
| Warm server decode time per proposal | 22.70264 ms | 21.68812 ms |
| Warm drafts accepted / proposed | 1,449 / 1,621 | 1,429 / 1,642 |

Each mode ran one cold request followed by one same-slot warm request. Both warm requests reused 79,996 prompt tokens and evaluated four. Both arms used the same mapped binaries, all eight routed experts, 42 of 42 layers offloaded, Sparse R2 attention and split MTP with one draft token. The only setting changed was the Q8_1 switch. Both warm logs sampled T2 launches and long-context R2 at query widths one and two. The enabled arm reported successful admission to the same 72-block grid. Diagnostic CUDA events were disabled. The throughput outputs first differed at token 27 in both comparisons. The performance fixture had no semantic score.

## Independent output checks

The enabled path passed a separate 2K JSON arithmetic test with the exact expected result through EOS. On an independent 80K input it returned the expected fact planted at token 40,000, with all seven tokens through EOS matching the earlier split-MTP recall. The recall process logged Q8_1 admission and a T2 launch sample, and the request logged long-context R2 and draft acceptance. Its short completion did not yield a request-local T2 sample. These two scored fixtures have separate inputs from the performance comparison. [Prior midpoint recall](../split-32k-recall-20261118/README.md)

## Reuse the published CUDA patch

Build the published Q8_1 Phase A patch on its pinned Sparse R2 source and use the cache-only split-MTP runtime. Keep `GGML_CUDA_OCCAMY_GDN_T2=1` and `GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0` in both arms. Set `GGML_CUDA_OCCAMY_GDN_T2_Q8A=0` for the control and `=1` for the candidate. The [allowlisted receipt](evidence.json) records the model-independent arguments, numeric environment, measured runtime hashes and timing definitions. The short verifier timing and this whole-request comparison use the same CUDA binary. [Published Q8_1 patch and reconstruction](https://github.com/average-user-887/llamaxwell-35/blob/7d35f484e55fc4435e5bc99e66b27cbcda0e35e4/gdn-q8-phase-a-20261118/README.md)
