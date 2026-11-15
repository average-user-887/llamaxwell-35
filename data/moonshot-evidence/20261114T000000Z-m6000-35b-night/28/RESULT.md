# Card 28 result: no prefill variant shipped, no N3 A/B run

| QK variant (one KV group: 512 cols, d 256) | 30K keys ms | 61K keys ms | 92K keys ms | vs TN default |
|---|---|---|---|---|
| TN default (current: sgemm_128x128x8_NT_vec) | 1.669 | 3.275 | 4.846 | 1.00 |
| NN, K transposed (QKNN idea) | 1.665 | 3.326 | 4.913 | 0.99-1.00 |
| NN, strided-batched 4096-key splits | 1.674 | 3.005 | 4.770 | 1.00 / 1.09 / 1.016 |
| transposed output S^T = Q^T K | 1.705 | 3.381 | 5.148 | 0.94-0.98 |
| best cublasGemmEx algo (NN algo 5/6) | 1.636 | 3.222 | 4.833 | 1.00-1.02 |

Numerics: TN and NN give bit-identical scores (max diff 0). RMSE vs FP64 is 2.9e-7 on unit-RMS scores for both.
Source: qkbench/result.txt and qkbench/nvprof.txt (two GPU rows of about 10 s each, 17:21 and 17:22, under the lock).
The 61K batched row (+9%) does not show up at 30K or 92K. With timing noise of about 3% at a 1.1 GHz
clock, it is not a dependable lever. At most it is worth ~0.5-2% of a deep ubatch.

Softmax (9% at 92K) runs at ~78% of memory bandwidth (242 GB per deep ubatch), so it is not pass overhead.
Neither cheap lever clears the bar. The gain that remains is removing the score round-trip: ~1.5 s of
traffic per deep ubatch out of 7.5 s of attention. That needs a fused flash-style kernel, which is
card-13 scope.

Decision: per card step 2, no op qualifies with clear headroom. Branch night15-prefill28 was not created
and the prod libs are unchanged. The GPU is free for the V-side Q8 KV row.
