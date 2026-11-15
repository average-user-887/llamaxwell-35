# Card 28c: fused exp-on-load PV microbench (failed the accept, no integration)

These are the best rows per shape, from the pvexp binary built from the v2 source (BK=8, split-K sub-partition, padded Bs). Each shape has 2 runs. One KV group: a 64-query tile x 8 heads = 512 cols, D=256, 4096-key splits.

| keys | sub | current ms (run0/run1) | fused ms (run0/run1) | fused/current | pvexp vs cuBLAS batched | RMSE vs FP64, current / fused |
|---|---|---|---|---|---|---|
| 92160 | 3 | 6.362 / 6.384 | 6.331 / 6.279 | 0.995 / 0.984 | 82.4% / 82.4% (4.07-4.11 TF vs 4.94-4.99) | 2.53e-08 / 1.47e-08 |
| 61440 | 2 | 4.564 / 4.326 | 4.192 / 4.204 | 0.918 / 0.972 | 86.1% / 86.4% (4.18-4.22 TF) | 2.76e-08 / 1.61e-08 |
| 30720 | 6 | 2.174 / 2.175 | 2.011 / 2.015 | 0.925 / 0.926 | 90.2% / 97.3% (4.23-4.39 TF) | 4.93e-08 / 2.90e-08 |

- **Accept (fused ≤ 0.90x current on two runs): FAIL** at all three shapes. The best is 0.92-0.93.
- **Cut line (≥ 83% of sgemm): reached** at 61K and 30K, and 82.4% at 92K. So the card ran to the accept test and failed it there.
- **Error:** fused RMSE is about 0.6x the current path at every shape, so it is inside the gate. But the accept failed, so this does not matter.
- No branch, worktree or prod change. Nothing was integrated.

## Why it cannot pass as built
- current = softmax with P write (1.48 ms at 92K) + cuBLAS batched NN (4.74-4.80 ms) + reduce.
- fused = read-only stats pass (0.71 ms) + pvexp (5.75-5.81 ms) + reduce.
- The fused path saves only the P write and re-read, about 0.75 ms at 92K. It then loses about 1.0 ms because pvexp, written in nvcc C++, is about 82% as fast as cuBLAS (cuBLAS is SASS-tuned).
- The stats pass reads all of S once, at about 266 GB/s (the memory bandwidth limit). It stays as long as exp needs the column max before the GEMM.
- To reach 0.90x at 92K, the GEMM must match cuBLAS (about 4.9 ms). Either that, or the max must come for free from the QK producer, or from an online rescale inside the GEMM.

## Variants tried (all 2 runs per shape)
| variant | 92K pvexp speed | 61K | 30K | note |
|---|---|---|---|---|
| v1: BK=8, 1 block per split | 77.8-78.3% | 76-81% | 70-74% | wave tail: 176 blocks / 48 slots at 92K, 56 at 30K |
| v2: + split-K sub 2/3/4/6, Bs row pad +4 | 82.4-82.7% | 81-86% | 79-97% | best; table above |
| v3: v2 with BK=16 | 76-81% | 72-84% | 63-94% | 68 B spill at 128 regs; worse |

- The current path itself varies by ±5% between sub sweeps (6.3-7.2 ms at 92K). This is clock warm-up drift. The ratios above use the current-path time measured in the same row.
- Synthetic inputs: S = 2·N(0,1), V = N(0,1), causal-tail half mask, full splits only. The FP64 reference is sampled on every 37th column.

## Possible follow-ups (not done tonight)
- Online max in pvexp (flash-style rescale every 64 keys). This removes the 0.71 ms stats pass. If the GEMM stays at about 82%, the 92K estimate is about 0.88-0.90x, which is marginal.
- Emit the per-split column max from a custom QK kernel. That needs a QK kernel at cuBLAS parity, which card 28 found is capped at about 5 TF.
