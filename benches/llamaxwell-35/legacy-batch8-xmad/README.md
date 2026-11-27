# Legacy batch8 MoE vocabulary head on Maxwell

The legacy Occamy 1.0 Q4_0 headq40b runtime was tested without MTP on a Quadro M6000. Both runtime variants use the same model, executable, host library, and two 98,304-cell slots. For a cached 2,048-token prompt with the second slot idle, the single-column XMAD vocabulary head increased generation **0.12–0.72%** and reduced whole-request time **0.13–0.72%**.

| Run order | 1,024-token warm round | OFF tok/s | ON tok/s | Generation change | Wall time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF → ON | 1 | 87.926 | 88.030 | +0.119% | −0.129% |
| OFF → ON | 2 | 87.864 | 88.034 | +0.193% | −0.165% |
| ON → OFF | 1 | 87.748 | 88.382 | +0.723% | −0.721% |
| ON → OFF | 2 | 87.682 | 88.264 | +0.663% | −0.658% |

All four warm pairs matched sampled P0 graphics and memory clocks at 1,151 and 3,304 MHz. The forward order fell below the 0.5% warm generation target. Cold wall time increased by 0.01% and 0.22%, with no matched sampled prefill clocks.

All 25 requests and 10,624 generated token positions matched across corresponding runs. The original library served a 64-output cold reference. Four fresh processes then ran the candidate library with XMAD OFF, ON, ON, OFF, repeating the cold case and two 1,024-output warm requests. The original, first OFF and second ON processes also served both slots concurrently, with 96-output cold and 256-output warm requests per slot. Both slots were observed active together. Token IDs, streamed text, output and cache counts, prompt counts and stopping fields matched. The compared requests used deterministic generation.

The legacy patch body matches the published SM52 N1 XMAD source change line for line including context. The legacy link replaced one CUDA object, matched the original CUDA library when relinked without the patch and retained the dynamic ABI. With that exact candidate library installed, set `GGML_CUDA_HEAD_ROWPACK_SM52=1` and `GGML_CUDA_HEAD_ROWPACK_N1_XMAD_SM52=1`. The guarded kernel handles the 248,320 × 2,048, single-column rowpacked Q4_0 vocabulary head on SM52. [Public source](https://github.com/average-user-887/llamaxwell-35/commit/ff82e875cda90dc06ee8371a460f05cf2b7703e9) · [Aggregate timings and artifact checksums](receipts.json)
