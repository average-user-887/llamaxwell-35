# Dual-slot cold validation of Sparse R2 with cache-only MTP

On one Quadro M6000, the Sparse R2 MTP1 stack with cache-only catch-up completed two simultaneous 32,768-token requests in **203.999 s**. The same Sparse R2 MTP1 configuration without cache-only catch-up took **215.067 s**. That is an **11.068 s (5.15%)** cold-wall reduction on this matched workload. Full attention without MTP completed in **199.743 s**.

| Configuration | Cold pair wall (s) | Aggregate stream decode (tok/s) | Decode overlap (s) |
|---|---:|---:|---:|
| Full F16, no MTP | 199.743 | 63.126 | 5.927 |
| Full F16, MTP1 | 213.890 | 63.590 | 5.772 |
| Sparse R2, MTP1 | 215.067 | 64.725 | 5.649 |
| Sparse R2, MTP1 plus cache-only catch-up | 203.999 | 64.835 | 5.641 |

Each arm ran one concurrent pair with two fixed prompts, zero prompt cache reuse, and 192 generated tokens per slot. The stacked arm matched all 192 output IDs in order for each slot against the unstacked sparse MTP1 arm. Both MTP1 arms accepted 85/105 drafts in slot 0 and 90/101 in slot 1. The comparison with full no-MTP is a wall and throughput reference. [Allowlisted evidence](evidence.json)

Cold pair wall runs from the first request start to the last request finish. Aggregate stream decode uses 383 output intervals over the time from the first to the last emitted token across the pair. The similar aggregate rates show that the cold-wall reduction in this comparison came chiefly before generation. All arms offloaded 42/42 layers. The stacked server mapped R2 CUDA and cache-only C++ libraries whose hashes matched the published build receipt. It emitted cache-only catch-up and Sparse R2 query-width 1 and 2 markers. Those markers establish use within the arm. Per-slot dispatch coverage remains unmeasured.

## Related 80K midpoint recall

A second paired recall test placed a fact beginning at token index 40,000 of the same 80,000-token prompt in each arm. Full F16 MTP1 and Sparse R2 MTP1 both returned the planted fact exactly. All seven emitted IDs, including EOS, matched across the two arms. This tests one midpoint. The earlier paired retrieval pass and synthetic distant-needle miss remain in the sparse evidence. [Sparse R2 evidence](../sparse-r2-20261117/README.md)

## Related single-slot warm 1,024-token replay

| Configuration | Request wall (s) | Emitted decode (tok/s) | Draft accepts |
|---|---:|---:|---:|
| Sparse R2, no MTP | 13.525 | 76.471 | — |
| Sparse R2, MTP1 plus cache-only catch-up | 13.212 | 79.086 | 479/544 |

Each arm reused 79,996 of 80,000 prompt tokens and emitted 1,024 tokens. Each arm's first 192 generated IDs matched its own prior warm reference. The two arms produced different outputs, and this fixture has no semantic score. On this one warm pair the stack cut request wall by 0.313 s (2.31%). The decode rate divides emitted tokens by the server's decode duration. The dual-slot table uses client stream timestamps and a different rate calculation.

## Source and fresh build

The existing stack recipe at the published revision reconstructs the tested source from the published Sparse R2 and cache-only patches. A fresh, offline Release build of that source linked the server, backend test executable and CUDA library, which contained 147 native `sm_52` images. The server version check passed, and CUDA relocations resolved with the toolkit link stub. Fresh binary hashes differ from the measured overlay. A live CUDA driver and fresh-binary GPU execution remain unverified. [Published stack recipe at 3eea327](https://github.com/average-user-887/llamaxwell-35/blob/3eea327fb4c289a54a8796899080bbbd1a3cdcfd/stacked-mtp-20261118/README.md)
