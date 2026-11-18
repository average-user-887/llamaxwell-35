# Q8_1 GDN with a 2,048-token microbatch at 80K

On the same Q8_1 CUDA build, changing `-ub 1024` to `-ub 2048` reduced cold prefill by **11.117 seconds** and cold request wall time by **11.029 seconds**. The warm request took **0.805 seconds longer**. The sum of the cold and warm request times fell by **10.223 seconds**, from **303.983 to 293.760 seconds**.

| Measurement | `-ub 1024` | `-ub 2048` |
| --- | ---: | ---: |
| Cold prefill | 265.86218 s | 254.74492 s |
| Cold request wall, 192 outputs | 268.10631 s | 257.07758 s |
| Cold decode throughput | 86.37358 tok/s | 83.41396 tok/s |
| Warm request wall, 3,072 outputs | 35.87659 s | 36.68194 s |
| Warm decode throughput | 86.26331 tok/s | 84.37456 tok/s |
| Sum of the two request wall times | 303.98290 s | 293.75952 s |

The reference is the earlier Q8_1-enabled run at `-ub 1024`. Each setting served one cold and one same-slot warm request with the same 80,000-token input. Both warm requests reused **79,996** prompt tokens and evaluated **four**. Both arms used all eight routed experts, 42/42 GPU offload, the same mapped libraries, and split MTP1. The larger microbatch was observed in both the target and draft contexts, with a 2,048-row cache-only dispatch during cold prefill. Both warm intervals sampled T2 launches and long-context Sparse R2 at query widths one and two. Both outputs first differed at token 45 across settings. This performance fixture had no semantic score. [Earlier Q8_1 warm comparison and reference](../gdn-q8-warm80k-20261118/README.md)

## Independent midpoint recall

A separate 80K request with the larger microbatch returned the expected fact planted at token 40,000. All seven output tokens through EOS matched the earlier stock split reference, Q8_1 with `-ub 1024`, and stock split with `-ub 2048`. Live logs confirmed Q8_1 admission, a process-level T2 sample, 2,048-row cache-only prefill and long-context Sparse R2 q1/q2. The seven-token request yielded no request-local T2 sample. The score covers one planted fact on a separate input.

## Reuse the published configuration

Use the published Q8_1 CUDA patch with the same Sparse R2 and cache-only split-MTP stack. Keep `GGML_CUDA_OCCAMY_GDN_T2_Q8A=1` and `GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0`. Change only `-ub 1024` to `-ub 2048`. Both runs used `-b 2048` and one 84,224-token slot. The [allowlisted receipt](evidence.json) contains the model-independent arguments, runtime flags, request geometry and measured timing. [Published Q8_1 CUDA source and reconstruction](https://github.com/average-user-887/llamaxwell-35/blob/7d35f484e55fc4435e5bc99e66b27cbcda0e35e4/gdn-q8-phase-a-20261118/README.md)
