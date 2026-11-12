# Independent mixed-serving measurement audit

CPU-only review, 2026-11-12 15:43 CET. No model, GPU, service or routing calls; no edits to the root measurement harness. Inputs reviewed: `model_mixed.py`, `serving_probe.py`, both `mixed-*.json` files and saved server logs/identity records under `model-runs/batch2048-a3` and `model-runs/batch1024-b1`.

## Accepted observations

All four runs have exactly 90,000 cached deep-prompt tokens plus one evaluated token; 512 deep output tokens; a fresh 16,384-token prompt with zero cache reuse; and 64 fresh output tokens. `truncated` is false for both requests in every run. The saved server logs confirm slot 1 restoration and slot 0 erasure, followed by the same native prompt/cached counts. Deep and fresh prompt strings match across all arms; the source token slices share zero leading tokens. There is no evidence of accidental truncation, prefix sharing or a falsely cold fresh prompt.

The fixture contains 788,960 tokens. Its current file SHA256 is `7766ffc2a3a95ae8d7740de9b9a41a9c505498bf2a357bfc91117ee0afde4bcb`. The source-text hash still matches the retained manifest (`7881b8bc5c046bcf06dfb50fb79a7702bf1d70ac2d738ec625d3ef59df17eea9`). Tokenization uses the pre-existing chunk convention; this is a consistent fixture, not proof that chunk tokenization equals one whole-document tokenizer call.

The two arm identities have the same image, GPU environment and CUDA library hash; their saved argument vectors differ only in container name and logical batch size. Physical microbatch remains 1024.

| Metric, mean of two requests per arm | Logical batch 2048 | Logical batch 1024 | Change |
|---|---:|---:|---:|
| Worst deep inter-token gap | 5.1938 s | 2.8342 s | -45.43% |
| Prefill-conditioned p95 gap | 5.0791 s | 2.7525 s | -45.81% |
| Fresh time to first token | 36.1054 s | 36.7902 s | +1.90% |
| Fresh request wall time | 38.1069 s | 38.8042 s | +1.83% |
| Combined makespan | 46.2706 s | 46.8547 s | +1.26% |

Fresh TTFT exceeds the server's prompt-processing timer by only about 18.5 ms. The deep stream remains active throughout fresh prefill, so the observed multi-second stalls represent actual coexistence. This supports a latency/throughput tradeoff, not a throughput improvement.

## Material limits and defects

1. Outputs repeat exactly within each arm but differ across logical batch sizes, despite equal prompts and greedy sampling. The fresh continuation first differs at character 13; the deep continuation first differs at character 594 (`end_of_paragraph` versus `end_of_sequence`). This is consistent with batch-shape numerical sensitivity, but the mechanism and quality consequence are not established. Do not describe the change as bitexact or quality-neutral.
2. Output token-ID arrays are empty. The evidence proves equal output counts and within-arm text equality, not token-ID equality. The native endpoint supports `return_tokens: true`; the server stores those IDs in the final response.
3. `serving_probe.py:128-130` retains only gaps wholly inside the measurement window. A gap crossing its start or end is omitted. Here the omitted gaps are only 18-95 ms, and the worst stalls remain included, so the main result survives. For a general prefill-stall metric include intervals that intersect the prefill window and separately report their clipped overlap.
4. All-stream p95 hides the stall problem because there are 511 deep gaps but only 9/17 wholly inside prefill. Prefer the maximum and prefill-conditioned distribution; the latter remains a small, correlated sample rather than an independent p95 population estimate.
5. Two repeated requests within one server start per arm do not provide independently replicated confidence intervals or thermal/order controls. Returning to the baseline arm would strengthen attribution before promotion.
6. The current harness checks stream completion but does not enforce expected predicted counts, `truncated == false`, cache counts or returned prompt dimensions before declaring success. Those checks pass in the inspected records, but should be explicit for future acceptance.
7. Timers intentionally exclude model loading, slot restoration and erasure. TTFT is meaningful for this admitted, prepared state, not whole-agent end-to-end latency.

## Integration caveat

When `BenchRoute` uses a rebuilt runtime, its executable and backend search path resolve under `/candidate`, while the supplied CUDA library is separately mounted under `/app`. The current scheduler runtime contains the exact same approved BUNDLE3 CUDA bytes, so this does not invalidate these results. If a different CUDA candidate is later combined with that runtime, mount it over `/candidate/libggml-cuda.so` too and record the actually loaded artifact rather than only the requested library path.
