# Split draft re-evaluation on a 3,072-token warm replay

On a single Quadro M6000, switching the Sparse R2 MTP1 stack from merged to split draft re-evaluation reduced the warm request from **38.929938 s to 37.518098 s** and raised decode from **79.463686 to 82.475794 tok/s (+3.79%)**. The fresh arms produced all **3,072 output IDs** in the same order. The split arm saved **1.411840 s** on this cached replay.

| Fresh 80K arm | Emitted decode (tok/s) | Warm request wall (s) | Draft accepted/proposed |
|---|---:|---:|---:|
| Split MTP1 | 82.475794 | 37.518098 | 1,449/1,621 |
| Merged MTP1 | 79.463686 | 38.929938 | 1,447/1,623 |
| Sparse no-MTP | 76.416306 | 40.337418 | — |

Fresh sparse no-MTP measured **76.416306 tok/s** and **40.337418 s**. The split arm gained **7.93%** in decode rate and saved **2.819321 s (6.99%)** in warm wall. The fresh no-MTP and split outputs first differed at token 27.

Each fresh arm ran a cold 192-output seeding request followed by one 3,072-output replay of the same 80,000-token prompt on the same slot. Every replay reused 79,996 prompt tokens and evaluated four. All used ubatch 1,024 with context shift and phase profiling disabled. The requests completed without truncation. Decode divides the emitted token count by the server's decode duration. Wall uses the complete streamed request. The fixture has no semantic score. [Allowlisted fresh measurements](evidence.json)

## Archived controls from the earlier 80K replay

| Archived N18 arm | Emitted decode (tok/s) | Warm request wall (s) | Draft accepted/proposed |
|---|---:|---:|---:|
| Merged MTP1 | 79.421957 | 38.945776 | 1,447/1,623 |
| Sparse no-MTP | 76.659728 | 40.206585 | — |

The split and fresh merged runs each reproduced all 3,072 output IDs of the archived merged run. These archived arms provide separate historical comparisons. The fresh comparison above supplies the matched controls for this follow-up. [Earlier long-replay evidence](https://github.com/average-user-887/llamaxwell-35/blob/58e05650146c75b1ff5ff27f09f70e52b1c5a27b/stack-prefill-20261118/README.md)

## Reproduce the setting

Use the previously published Sparse R2 plus cache-only MTP stack. Keep the model, the one-slot 84,224-token context, batch size 2,048, microbatch and mixed batch size 1,024, single-stream prefill, and MTP1 with F16 draft K/V. Keep `--no-context-shift`. Both MTP arms use `LLAMA_MTP_CACHE_ONLY=1`. For split, change `LLAMA_MTP_MERGE_REEVAL=1` to `LLAMA_MTP_MERGE_REEVAL=0` as the sole runtime environment change. Use the published sparse-only overlay and no MTP specification for the no-MTP arm. [Stack reconstruction](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md) · [Original split setting and phase evidence](https://github.com/average-user-887/llamaxwell-35/blob/ee5e368dd9b181f3ffbccbea8883dfca33e10972/split-draft-20261118/README.md)

Issue both requests with `cache_prompt=true`, seed 7 and temperature zero. Stream the replay, keep EOS ignored for this fixed-output fixture, and cap the replay at 3,072 tokens. The recorded per-arm deadline allowed up to 420 s with a 600 s outer wrapper. The same server executable and mapped library hashes were observed in both fresh MTP modes, and all three modes offloaded 42/42 layers. Warm Sparse R2 dispatch was observed at q2 for both MTP modes and q1 for no-MTP. Total dispatch coverage remains unmeasured.
