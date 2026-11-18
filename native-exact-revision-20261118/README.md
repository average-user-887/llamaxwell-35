# Native learned MTP on an exact 80K code revision

On one 80,000-token code revision task, native one-token MTP raised cached decode throughput from **76.024 to 88.392 tok/s (+16.27%)** and cut cached request wall time by **3.021 seconds (13.26%)**. Both modes returned the same 1,722 tokens through EOS. Cold decode was also faster, but the cold request took 0.667 seconds longer with MTP.

| Measurement | Sparse R2 with speculation off | Sparse R2 with native split MTP |
| --- | ---: | ---: |
| Cold request wall | 285.164 s | 285.831 s |
| Cold server time to first token | 262.475 s | 266.221 s |
| Cold decode throughput | 75.972 tok/s | 87.926 tok/s |
| Cached request wall | 22.780 s | 19.759 s |
| Cached decode throughput | 76.024 tok/s | 88.392 tok/s |
| Learned draft acceptance, cold and cached | — | 861/861 each |

There was one cold request and one same-slot cached request per mode. Both cached requests reused 79,996 tokens and evaluated four. Decode throughput divides emitted tokens, including EOS, by server decode time.

All four measured outputs and both archived full-attention F16 serial outputs matched token for token through EOS. Each response also matched the expected complete Python source after trimming outer whitespace and matched its parsed Python AST. The task was a source-preserving edit to a file from a pinned public llama.cpp revision. The result covers one revision task and one input.

## Reproduce the measured settings

Use the existing Sparse R2 and cache-only source stack, with `LLAMA_MTP_MERGE_REEVAL=0` and `LLAMA_MTP_CACHE_ONLY=1` in both arms. The same server executable and mapped libraries were used. Use the common settings in [evidence.json](evidence.json), including one 84,224-token slot, `-b 2048 -ub 1024`, no context shift, a 3,072-token output cap, and greedy sampling. Submit the same input twice to slot zero in each arm.

For the serial arm, append `--spec-type none`. For the learned MTP arm, append `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16`. These speculation arguments were the sole server configuration difference in the matched comparison. Name the model and place the corresponding shared libraries according to the build recipe.

- [Reconstruct and build the source stack](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md).
- [Review the same fixture's full-attention F16 serial reference and separate prompt-lookup profile](https://github.com/average-user-887/llamaxwell-35/blob/b11fe17cd514b3983c7c5685bba239ad5fed56d0/lookup-80k-20261118/README.md).
