# Independently checked root-run GPU results

Analysis completed 2026-11-12 at 15:58 CET. Inputs are the root's retained `recompute/gpu-98k-64-4096.jsonl` and `online/gpu-98k-64-4096.jsonl`, plus the earlier packed-GQA v2 receipt. No GPU calls were made by this reviewing worker. All binaries remain frozen at the hashes in `SHA256SUMS`.

## Measured operator results

Shape is 1024 queries, 98,304 keys, 16 query heads, two KV heads, dimension 256; packed tile has 512 columns and split-K is 4096. Each variant has three ABBA blocks in one process, six timings per arm, and the same one deterministic input seed. Candidate timings cover the whole attention algorithm, including conversion, packing, QK, softmax, PV, reduction and submission gaps.

| Variant | Reference mean | Candidate mean | Time reduction | Throughput ratio | Softmax mean |
|---|---:|---:|---:|---:|---:|
| Original packed-GQA v2, earlier run | 674.654 ms | 457.237 ms | 32.23% | 1.476x | 147.864 ms |
| Recompute exponents | 669.210 ms | 425.853 ms | 36.36% | 1.571x | 115.523 ms |
| Online maximum/sum | 669.624 ms | 406.248 ms | 39.33% | 1.648x | 90.730 ms |

Recompute's candidate range is 424.586-427.185 ms; online's is 405.541-407.843 ms. Per-block time reductions are 36.28-36.47% and 39.07-39.55%, respectively. These block ranges describe repeatability within these runs, not independent-session confidence intervals.

Compared with the earlier v2 candidate mean, recompute uses 6.86% less total time and online uses 11.15% less. Softmax time decreases by 21.87% and 38.64%. These incremental comparisons are cross-run/unpaired; they are not direct randomized v2-versus-variant experiments. Reference time also differs slightly across runs.

## Numerical result

Both root-run comparisons cover 4,194,304 FP32 output elements and pass the unchanged numerical gate:

| Variant | Nonfinite | Maximum absolute error | RMSE | Different bit patterns versus reference |
|---|---:|---:|---:|---:|
| Recompute | 0 | 3.52971255779e-7 | 2.55410351756e-8 | 4,172,499 |
| Online | 0 | 3.52971255779e-7 | 2.55425847197e-8 | 4,172,605 |

Maximum relative error with the existing `1e-5` denominator floor is 0.00985687620414 for both. These results are not bitexactness or model-quality proof. The recompute aggregate error metrics match v2, but equal aggregate metrics alone do not prove its complete output buffer is identical to v2.

Explicit scratch remains 637,552,640 bytes (608.018 MiB). Reported post-first-call free-memory delta is 649,068,544 bytes (619 MiB), not transient peak allocation. Model integration may have different memory pressure and allocator interactions.

## Actual-mask and all-masked-row review

The synthetic kernels deliberately retain v2's contiguous causal-prefix assumption: `last_key = key_count - query_count + query_start + local_query`. They have no additive-mask parameter. They are therefore not drop-in arbitrary-mask model kernels, despite the successful synthetic comparison.

If the current online prototype were simply fed pre-masked logits and every key were `-inf`, each thread's sum would stay zero, the block inverse would become infinity, and final `0 * infinity` would produce NaNs. Mixed finite/`-inf` rows can be handled by the stable recurrence; entirely masked rows need an explicit zero/fallback policy. NaN or positive-infinity logits/masks remain outside the validated domain.

The prefill worker's current integration kernel at `prefill/gemm/integration/src/ggml/src/ggml-cuda/fattn-maxwell-gemm.cuh:80` already reads actual additive masks using the supplied byte row stride, handles padded query rows, and uses `total > 0 ? 1 / total : 0`. That is stronger than the synthetic prototype and must be preserved in any later online replacement.

Advisory sent to the prefill worker, without editing their source:

1. Keep the padded-query uniform zero/return and real `mask_row_stride` addressing.
2. Iterate the complete actual key dimension, not a derived synthetic causal bound.
3. Use `score + additive_mask` in the online first pass and again in final exponent normalization; finite biases matter, not only binary masking.
4. Keep the finite `-FLT_MAX` initializer, rescale each thread's sum by its local/global maximum difference, then reduce the sums.
5. Explicitly write zero for a zero-denominator row before normalization. Do not rely on multiplying a potentially undefined exponential by zero.
6. Retain existing dispatch exclusions for sinks, softcap and unsupported bias/mask layouts; test arbitrary finite bias, holes, padded rows and all-masked rows before model use.

## Acceptance boundary

Online is the preferred next softmax candidate for a separate masked-model integration experiment. The measured result is an isolated full-attention-operator improvement at one shape and seed, not a demonstrated end-to-end serving or agent speedup. Allocation/setup/input transfers remain outside the operator timer. Real activations, irregular shapes, actual masks, model quality, serving memory headroom and deployment remain separate root-owned acceptance work.
