# Card 28 - prefill op profile (batch8 libs, single slot, nvprof)

Method: batch8 binaries ($N2/9/bin) with the prod GGML_* env and prod server args (-np 2, -c 196608,
-b/-ub 2048, mixed-batch 1024, prefill-single-stream, FA on), wrapped in `nvprof --print-gpu-trace`.
One 94,208-token prompt (d0 fixture), n_predict 1. Kernels classified by name (an28.py); ubatches cut
every 30 GDN launches. ms = GPU kernel time per 2048-token ubatch. nvprof serialises kernels, so
absolute t/s here is below the un-instrumented two-slot replay; shares are what matter.
Evidence: prof/a-20261115T160951Z (trace.csv, trace.csv.segs.json, server.log).

| KV depth | total ms | t/s (wall) | attention | MoE GEMM (mmq) | dense proj (dequant+SGEMM) | GDN | norms | rope/KV/other |
|---|---|---|---|---|---|---|---|---|
| 0      | 3130  | (warm-up) | 172 (6%)   | 1354 (43%) | 1169 (37%) | 236 (8%) | 108 (3%) | 89 (3%) |
| 12K    | 4214  | 484 | 1163 (28%) | 1398 (33%) | 1212 (29%) | 238 (6%) | 111 (3%) | 92 (2%) |
| 30K    | 5735  | 356 | 2665 (46%) | 1408 (25%) | 1218 (21%) | 239 (4%) | 111 (2%) | 93 (2%) |
| 61K    | 8152  | 250 | 5076 (62%) | 1411 (17%) | 1219 (15%) | 240 (3%) | 112 (1%) | 93 (1%) |
| 86K    | 10149 | 201 | 7064 (70%) | 1415 (14%) | 1225 (12%) | 239 (2%) | 112 (1%) | 94 (1%) |
| 92K    | 10553 | 192 | 7463 (71%) | 1421 (13%) | 1225 (12%) | 239 (2%) | 111 (1%) | 94 (1%) |

Attention split at 92K (ms / share of ubatch): QK cuBLAS `sgemm_128x128x8_NT_vec` 3302 (31%),
PV `maxwell_sgemm_128x64_nn` 3080 (29%), fused split softmax 980 (9%), K/V fp16->fp32 convert 44,
reduce 45. At 30K: QK 1136, PV 1143, softmax 338.

## Achieved fraction of bound (M6000: ~1.12 GHz measured under load, 3072 FP32 lanes -> ~6.9 TFLOPS FP32; 317 GB/s)

| op (92K) | work | time | achieved | bound type |
|---|---|---|---|---|
| QK (10 FA layers) | 1.58e13 FLOP | 3.30 s | 4.8 TFLOPS = ~69% | compute (FP32, no fast fp16/dp4a on sm_52) |
| PV | 1.58e13 FLOP | 3.08 s | 5.1 TFLOPS = ~74% | compute |
| softmax | 640 calls x 189 MB scores read+write = 242 GB | 0.98 s | 247 GB/s = ~78% of 317 GB/s | memory |
| MoE mmq | ~8 experts x 3 mats, 40 layers | 1.42 s | flat in depth, int8 mmq without dp4a | compute (int) |
| dense SGEMM | dequant + generic NT_vec sgemm | 1.23 s | flat in depth | compute |

## Verdict (updated after the step-2 microbench, qkbench/result.txt)
Attention is the only op with >=25% share at both windows (46% at 30K, 71% at 84-92K), but it has no cheap
headroom:
- QK: cuBLAS on sm_52 tops out near 5.0 TFLOPS (~72% of FP32 peak) for this shape whatever the layout.
  K transposed + OP_N/OP_N gives bit-identical scores at speedup 0.986-1.015. Every cublasGemmEx algo
  (0-23, TN and NN) and the transposed-output and 4096-split batched forms come in at <=5.0 TFLOPS. The best,
  NN batched-4096 at 92K, is 1.6% faster on QK alone, about 0.5% of a ubatch and within run-to-run noise.
  The QKNN flag is dropped.
- Softmax: an earlier draft of this file said it was pass overhead. That was wrong. It moves 242 GB per
  deep ubatch at ~78% of memory bandwidth. A leaner pass cannot remove that traffic.
- The real lever is to remove the score round-trip: QK write 189 MB, softmax read and write 377 MB, PV
  read 189 MB per call, ~1.5 s per deep ubatch. That needs a flash-style fused kernel (the card-13 attention
  rewrite), not a cheap flagged variant.
- MoE (13-25%) and dense (12-21%) are below the 25% line at depth. GDN and norms are small.
No op meets share >=25% plus clear headroom with a cheap variant, so no N3 A/B was run. The GPU goes to the
V-side Q8 KV row.
