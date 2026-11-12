# M6000 long-context prefill investigation

**12 November 2026 · Quadro M6000 / SM 5.2 · experimental, default off**

Packed grouped-query attention (GQA) using FP32 cuBLAS GEMMs reduces measured deep-append prompt time by **26–27%**, and a separate staged fresh-98K trajectory by **21.94%**, against an already heavily optimized baseline. These are **prompt-processing improvements, not decode tokens/second gains**.

**This is not a quality-equivalent replacement or a production recommendation.** At the 65,536-token checkpoint, the paired full-vocabulary distribution changes by **KL 0.293615 nats and total variation 20.8524%**, despite unchanged top-1 choices. That negative evidence remains important alongside the speed and semantic successes.

## What changed

The candidate packs the eight query heads sharing each KV head into wider matrix multiplications, then applies masked softmax and a split/reduced value product. It retains the full context, FP32 arithmetic, existing F16 KV storage and the actual additive attention mask. It changes arithmetic ordering and normalization placement; FP32 does **not** mean bit-identical to the baseline.

Version 4 stages one KV group at a time. Its fixed scratch allocation is **432,013,888 bytes, approximately 412 MiB**, saving approximately 192 MiB versus version 3 without changing tested output bytes. The observed v4 serving allocation was approximately 23,458 MiB, leaving 1,006 MiB free in the tested profile; this is not a universal memory guarantee. Scratch is per backend context/stream, not a global maximum.

The default-off switch is `GGML_CUDA_FA_MAXWELL_GEMM=1`. Dispatch is deliberately narrow: SM 5.2, F32 queries/output, F16 keys/values, head dimension 256, 16 query heads / 2 KV heads, one sequence, 64–1,024 queries and 4,096–98,304 keys. Unsupported shapes and features fall back; ordinary single-token and paired decode use existing kernels. Source review found no shared-scratch race under the server's current serial submission, but it is not a race-detector proof. See [source and lifetime review](prefill/CLOSEOUT_REVIEW.txt).

## Measured performance

The model was `occamy-1.0-Q4_0-headq40b.gguf`, with all 41 layers offloaded to the GPU. Matched comparisons use the same candidate library with GEMM disabled or enabled; no CPU-only inference was substituted.

| Workload | GEMM off | GEMM on | Interpretation |
|---|---:|---:|---|
| 4,096-token append after a restored 90K prefix, production checkpoint settings | 31.925 / 31.949 s | 23.451 / 23.491 s | About 26.5% less prompt time; two repeats per arm |
| Fresh 98,000-token trajectory, six staged requests | 481.332 s | 375.744 s | 21.94% less summed prompt time; one paired trajectory |
| Fresh 95,167-token retrieval, staged requests | 459.387 s | 359.157 s | 21.82% less summed prompt time; one paired trajectory |

The append test restores a baseline-produced prefix and measures only the new append; it is not a fresh-prefix quality test. Other v3/v4 append repeats gave the same approximately 26–27% reduction. The fresh trajectories build their own state under each arm. Their totals are **not single-request time to first token (TTFT)** and exclude model loading and warmup.

The original fresh-98K and early mixed tests used matched checkpoint defaults of 32 checkpoints / minimum spacing 8,192. The later append repeat, quality grid and 95K retrieval explicitly use production checkpoint settings of 16 / 2,048. Cache RAM remains disabled for isolation, so even these are not every production setting reproduced exactly.

Evidence: [append summary](quality-analysis/deep-performance-summary.json), [repeat comparisons](quality-analysis/performance-repeat-analysis.json), [fresh-98K timing receipt](quality-analysis/quality98-v4-cache-timing.json), and the [95K off](model-runs/deep-retrieval-off-a1/retrieval-receipt.json) / [95K on](model-runs/deep-retrieval-on-b1/retrieval-receipt.json) receipts.

## Quality: successful checks and a material unresolved difference

### Full-vocabulary screen

The stratified grid contains **64 fixed teacher-token rows**, 16 at each of four context depths, plus **four paired full-vocabulary distributions**, each containing all 248,320 token IDs. Raw top-1 choices agree in **64/64** rows. The teacher token is top-1 in **59/64** rows in both arms; agreement is not 64/64 accuracy.

| Context position | KL(off \|\| on), nats | Total variation |
|---|---:|---:|
| 16,384 | 0.00737165529 | 5.673073% |
| 32,768 | 0.00022257677 | 0.083856% |
| **65,536** | **0.29361529283** | **20.852358%** |
| 98,000 | 0.00000576035 | 0.002092% |

Mean teacher negative log-likelihood changes from 0.2601810684 to 0.2549830636 nats, a small signed improvement of 0.0051980048. That average does not cancel the large distribution movement at 65K. These KL values use renormalized API probabilities, not original logits. One fixture, 64 labels and four distributions cannot establish general task quality, nor does this screen alone prove degraded task quality.

Independent 16K OFF/OFF and ON/ON controls reproduce every retained probability exactly; changing the baseline microbatch from 512 to 1,024 also preserves those probabilities. Thus the observed cross-arm drift is not explained by noise or that tested microbatch change. A separate deep-append greedy continuation first diverges at zero-based token 22. See the [complete grid](quality-analysis/QUALITY-GRID-V4.txt) and [determinism controls](quality-analysis/QUALITY16-CONTROLS.txt).

### Semantic and operator checks

| Check | Verified result | Limit |
|---|---|---|
| Copy-edit after approximately 17.9K tokens | Requested edit passes the independent AST check; all **397 completion token IDs**, including reasoning, and message content match | One task; expected-byte mismatch is only a final newline |
| Controlled no-thinking retrieval at 17,652 tokens | Both retrieve all four records; all **59 output token IDs** and content match | One fixture |
| Fresh retrieval at 95,167 tokens | Both retrieve all five records; all **88 output token IDs** and content match | One staged trajectory per arm, not a broad retrieval suite |
| Irregular attention probe, 1,023 queries / 98,048 keys | Additive masks, holes, padded strides, tails, all-masked rows and output canaries pass | Synthetic operator inputs, not model-quality acceptance |
| Sampled FP64 reference at model scale 0.0625 | Deep-case RMSE: baseline 2.7743e-8, v4 7.1581e-9; v4 closer on 90/112 ordinary sampled floats | Sparse synthetic sampling; does not erase real-model distribution drift |

The thinking-enabled retrieval baseline exhausted its 512-token budget during reasoning. That truncation is **not** evidence that the candidate has better retrieval quality; the controlled no-thinking comparison above is the appropriate paired result.

Evidence: [independent closeout review](prefill/CLOSEOUT_REVIEW.txt), [precision results and hashes](prefill/precision-audit/GPU_RESULTS.txt), [copy-edit off](model-runs/long-task-off-a1/copy_edit-0-response.json) / [on](model-runs/long-task-on-b1/copy_edit-0-response.json), [short retrieval off](model-runs/long-retrieval-nothink-off-a2/retrieval-0-response.json) / [on](model-runs/long-retrieval-nothink-on-b2/retrieval-0-response.json), and [95K retrieval off](model-runs/deep-retrieval-off-a1/retrieval-response.json) / [on](model-runs/deep-retrieval-on-b1/retrieval-response.json).

## Mixed scheduling: responsiveness, not a throughput multiplier

With two restored 90K slots, one request generates while another appends 4,096 tokens. A separate mixed-prefill cap reduces how long the competing stream appears stalled.

| Arm | Foreground TTFT | Maximum client-visible competing-stream gap | Combined wall time |
|---|---:|---:|---:|
| GEMM off, cap disabled | 32.200 s | 15.848 s | 42.086 s |
| GEMM on, cap disabled | 23.586 s | 11.578 s | 33.397 s |
| GEMM on, cap 1,024 | 23.792 s | 5.951 s | 33.664 s |
| GEMM on, cap 512 | 25.663 s | 3.236 s | 35.457 s |

These are means of two final repeats per arm, not a confidence interval or a broad service-level guarantee. Compared with GEMM alone, cap 1,024 reduces the visible maximum gap by **48.6%**, with approximately **0.87% more foreground TTFT and 0.80% more combined wall time**. Cap 512 reduces the gap by 72.0%, but costs 8.80% more TTFT and 6.17% more combined wall time. Thus 1,024 is the better measured compromise when near-unchanged total latency matters; 512 deliberately trades more throughput for responsiveness. These repeats use matched checkpoint settings of 32 / 8,192, not the production checkpoint profile. See the [final repeat analysis and receipt hashes](quality-analysis/deep-mixed-final-analysis.json).

**Retained observability failure:** the cap-1,024 strict receipt contains only 510 emitted token IDs for 512 server-generated tokens. Source and raw stream inspection identify UTF-8 fragment withholding, not parser loss. The derived analysis reports only client-visible timings, marks completeness false and leaves token identity unavailable. It does not reconstruct missing IDs or turn the strict failure into a pass. See [mixed-stream evidence and limitations](scheduler/analysis/DEEP-MIXED-CLIENT-VISIBLE.txt).

The final repeats explicitly use the separate client-visible-only contract: HTTP completion, server token counts, cache use and non-truncation are checked, but incomplete SSE token IDs remain labeled incomplete. Cross-arm token identity and general quality equivalence are not claimed. Gaps crossing the prefill-window boundary are included rather than discarded.

## What did not work, and where to look next

- **Occupancy-only attention tuning lost.** Tested tile/thread variants were approximately 40–72% slower. Higher occupancy did not compensate for reduced KV reuse.
- **Native n-gram speculation lost on the representative task.** An isolated fix removes ng8's output-head allocation cliff, but two matched repeats still take about **11.181 s versus 5.672 s**, approximately **97% slower**, despite 267/340 accepted drafts. The valid 391-token copy-edit output is identical. Draft acceptance alone is not a speed metric. The head fix is a rejected experiment, not part of the final GEMM runtime. See [ng8 audit](head9/NG8-AUDIT.txt).
- **Q8 KV is a capacity investigation, not a demonstrated speedup.** Stabilized decode microprobes are essentially flat: +0.39% single / -0.70% paired. Integrated-model quality, peak memory and two-slot 128K capacity were not measured. Simply changing cache flags would bypass current fast paths and still reserve F16 conversion storage; allocator, decode and prefill integration must be considered together. See [Q8 forward look](prefill/Q8_FORWARD_LOOK.txt).
- **Further attention work should preserve packed-GQA reuse.** A representative synthetic breakdown is approximately 161 ms QK, 150 ms value product, 91 ms softmax, 2.4 ms conversion and 1.5 ms reduction. Reducing score traffic or improving the value-product shape is more promising than conversion-only work. No additional multiplier is promised.
- **CUDA graphs fail with the current two-slot N2SPLIT path.** Graphs-off completes two 512-token-per-slot pairs in 16.554 / 16.537 s; graphs-on aborts because `ggml_cuda_occamy_gdn_split2` synchronizes the capturing stream while reading recurrent-state row indices. There is no graphs-on speed or quality result. Historical E07 replay evidence does not qualify BUNDLE3 N2SPLIT, and caching those mutable indices once would risk stale replay behavior. Keep the override absent; a real fix needs replay-safe device-indexed state gathering or correct content-aware graph updates. See the [current failure and source audit](scheduler/analysis/DEEP-GRAPHS-RESULT.txt) and [historical evidence](prefill/GRAPH-HISTORY.txt).

The next acceptance work is targeted numerical and semantic coverage around the 65K drift, followed by broader representative tasks. Successful retrieval and better sampled FP64 accuracy do not remove that requirement. No internet-wide speed ranking was established in this investigation.

## Closeout

GPU experiments ended at 17:32 CET, leaving the remainder of the authorized window for review and publication. The ownership lock is released, the previously active slotkeeper is restored, owned experimental containers and routers are gone, and only the two campaign-created slot copies were removed. User slot-file sizes and modification times match the recorded starting metadata. The original source checkout is clean and the production CUDA library hash is unchanged. The routed model is unloaded; the final GPU snapshot shows 0% utilization and no compute processes. No experimental build was deployed. See the [cleanup receipt](quality-analysis/FINAL-CLEANUP.json).

## Provenance and public package scope

| Artifact | Identity |
|---|---|
| Baseline BUNDLE3 commit | `2436f7554c0c8d70afc9acbcce85b2457193a516` |
| Tested v4 CUDA library SHA256 | `0c7ffbde334686f17f5ed6a1676c0cd966ec931eaa388900befb06af493330d5` |
| Complete v4 integration patch SHA256 | `77d992a5336019ee9804eae4102bb1784ed6d730672008ac6fc0a6b8db620da4` |
| Build toolchain | CUDA 12.8.93, SM 5.2 target, g++ 14 |

The owner authorized publication to the **public** `average-user-887/llamaxwell-35` repository on the separate `codex/maxwell-gqa-investigation-20261112` branch. The intended package contains the tested v4 CUDA and mixed-scheduler components, source patches, runtime binaries, sanitized selected receipts and this report. This report does not itself certify that an upload has completed. No pull request or main-branch change is implied, and publication is not deployment approval.

Excluded material includes model weights, saved user slot states, credentials, machine-control configuration, private conversation/rollout material and private backup artifacts. Machine-specific locations and identifiers are removed from public evidence. The publication manifest records selected files, archive members, transformations, and original/published hashes. Extract category and per-run archives into one common root to resolve this report's relative evidence links; the packaged runtime is under `runtime/`.

Archived harnesses retain their campaign-specific routing, ownership and deadline assumptions and are not unattended launch instructions. Published script copies are labeled `.archival.txt`; reproducing the experiment requires a newly authorized, bounded environment and an appropriate local adapter. The candidate remains **experimental and default off**.
