# Split MTP at 32K and 80K midpoint recall

On the matched 32,768-token Sparse R2 fixture, split MTP raised decode from **78.1830 to 83.6299 tok/s (+6.97%)** for a 1,024-token cached replay. Warm wall time fell from **13.19543 to 12.42057 s**, saving **0.77486 s (5.87%)**. The split and no-MTP outputs first differed at token 50 in both request conditions. The throughput fixture has no semantic score.

| 32K result | Sparse no-MTP | Sparse split MTP1 |
|---|---:|---:|
| Cold request wall (s) | 79.66256 | 81.17820 |
| Cold emitted decode (tok/s) | 78.6163 | 78.2732 |
| Cold accepted drafts | — | 81/110 |
| Warm request wall (s) | 13.19543 | 12.42057 |
| Warm emitted decode (tok/s) | 78.1830 | 83.6299 |
| Warm accepted drafts | — | 471/552 |

Each arm sent one 192-output cold seed and then one 1,024-output replay of the same 32,768-token prompt in the same slot. Both replays reused 32,764 prompt tokens and evaluated four, with context shift disabled. Within each mode the warm first 192 IDs matched its own cold seed. Both conditions also matched the first 192 IDs from their N14 reference. The prior no-MTP reference used the same mode. The prior MTP reference used merged re-evaluation. Cold decode fell 0.44% in the split arm, and cold request wall increased 1.51564 s. Decode divides emitted tokens by the server decode duration. [Allowlisted measurements](evidence.json)

## Separate 80K midpoint recall

The live N27 split configuration at ubatch 1,024 retrieved the planted fact starting at token index 40,000 of the 80,000-token prompt. The raw input IDs and rendered template matched both earlier N7 controls, full-attention MTP1 and Sparse R2 MTP1. All seven generated IDs including EOS matched both controls. The server used the pinned split runtime, offloaded 42/42 layers, and logged 1,024-row cache-only catch-up and long-context Sparse R2 q2 dispatch. This checks one planted fact. [Earlier midpoint reference](https://github.com/average-user-887/llamaxwell-35/blob/31f46e620ed3648a3af1b9cdd519b7174e720334/stack-validation-20261118/README.md)

## Reproduce the settings

Use the published Sparse R2 plus cache-only stack, with `LLAMA_MTP_MERGE_REEVAL=0` and `LLAMA_MTP_CACHE_ONLY=1`. The 32K arms used identical server binaries, mapped libraries, and environment settings. The split arm added `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16`. The control omitted those draft arguments. Both held one slot, 84,224-token context, batch size 2,048, microbatch and mixed batch size 1,024, single-stream prefill and disabled context shift. Use seed 7, temperature zero, prompt caching, and fixed output counts with EOS ignored on this throughput fixture. [Original split configuration](https://github.com/average-user-887/llamaxwell-35/blob/31f46e620ed3648a3af1b9cdd519b7174e720334/split-draft-20261118/README.md) · [Stack source reconstruction](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md)

The 80K midpoint recall uses the earlier uninstrumented N27 split settings and EOS stopping on its independent recall fixture. The published source supports these settings.
