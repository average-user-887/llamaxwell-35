# Native split MTP with a 2,048-token microbatch

On one 80K-token performance fixture, changing `-ub 1024` to `-ub 2048` reduced cold request wall time by **10.629 seconds**. Across the cold request and its same-slot cached follow-up, the change saved **10.455 seconds (3.43%)**. The cached request took 0.174 seconds longer with the larger microbatch.

| Measurement | `-ub 1024` | `-ub 2048` |
| --- | ---: | ---: |
| Cold request wall, 192 output tokens | 267.668 s | 257.039 s |
| Cold prompt evaluation | 265.310 s | 254.658 s |
| Cold decode throughput | 82.178 tok/s | 81.349 tok/s |
| Warm request wall, 3,072 output tokens | 37.518 s | 37.693 s |
| Warm decode throughput | 82.476 tok/s | 82.089 tok/s |
| Sum of the two request wall times | 305.186 s | 294.732 s |

Each setting contributes one cold request and one warm request. The `-ub 1024` sample is the previously reported N37 split arm. Both settings used identical 80,000-token inputs and reused 79,996 cached tokens while evaluating four on the warm request. Throughput divides emitted tokens by server decode time. The cold and warm outputs each first diverged at token 27 across settings, counting from one. The performance fixture had no semantic score.

## Reproduce the setting

Use the existing Sparse R2 and cache-only source stack and the split settings `LLAMA_MTP_MERGE_REEVAL=0` and `LLAMA_MTP_CACHE_ONLY=1`. Change only `-ub 1024` to `-ub 2048`. Keep `-b 2048`, `--mixed-batch-size 1024`, the one-slot 84,224-token context, and the one-token F16 MTP draft. The [allowlisted receipt](evidence.json) records the full model-independent performance argv, shared environment, request settings, and binary hashes. Live measurements confirmed that the target and draft each used a microbatch of 2,048. This setting uses existing source.

- [Reconstruct and build the source stack](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md).
- [Review the archived N37 split comparison](https://github.com/average-user-887/llamaxwell-35/blob/2fa4f548a787df2ff183f5531fa04580aa61196d/split-warm-3072-20261118/README.md).

## Independent midpoint recall

A separate run with native split MTP and `-ub 2048` returned the expected fact planted at token 40,000 of an 80,000-token prompt. All seven output IDs through EOS matched N50 split `-ub 1024`, both N7 references, and the N17 `-ub 2048` stack reference. The server reported 42 of 42 layers offloaded, 2,048 cache-only rows, and long-context Sparse R2 query widths of one and two. This checks one planted fact on the recall fixture.
