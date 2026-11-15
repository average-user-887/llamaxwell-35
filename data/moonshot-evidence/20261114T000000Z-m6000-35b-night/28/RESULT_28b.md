# Card 28b result: stopped at the microbench (gain about 1%, gate is 5%)

| keys | serial ms/tile (QK + softmax + PV + reduce) | 2-stream pipelined ms/tile | gain | max abs diff vs serial |
|---|---|---|---|---|
| 30720 | 3.963 / 3.964 | 3.925 / 3.947 | 0.96% / 0.41% | 0 |
| 61440 | 7.899 / 7.914 | 7.820 / 7.849 | 1.00% / 0.82% | 0 |
| 92160 | 11.497 / 11.477 | 11.375 / 11.338 | 1.06% / 1.21% | 0 |

Each cell gives two repeats. One tile is 64 queries x 8 heads for one KV group. The kernels are the same
as fattn-maxwell-gemm.cuh: cuBLAS TN QK, fused split softmax, batched NN PV and the LSE reduce.
Source: qkbench/pipe.cu, qkbench/resultpipe.txt and qkbench/nvprofpipe.txt (GPU row 17:26:02-17:26:10).

Variant tested: the cheapest way to hide the round-trip without new numerics. Scores are double-buffered
and the memory-bound softmax, PV and reduce for tile t run on a second stream while QK for tile t+1 runs.
The output is bit-identical (diff 0), so RMSE vs FP64 is unchanged. It gains only about 1%. The cuBLAS
SGEMM blocks fill every SM, so the softmax blocks cannot co-reside and run in the gaps between them.

At 92K, the per-tile split in the microbench is QK 4.85 ms, softmax 1.5 ms (13%) and PV 4.9 ms. The
other variants the card names all need our own GEMM:
- Fusing the softmax into PV operand loads needs a custom PV SGEMM that applies exp(s - m) on load.
- Writing P directly in the PV layout needs the same custom PV SGEMM.
- Fusing max/sum into the QK epilogue needs a custom QK SGEMM, because cuBLAS has no epilogue on sm_52.

To gain 5% of deep attention, a custom CUDA-C PV kernel with exp-on-load would have to reach at least
~83% of the cuBLAS SASS `maxwell_sgemm_128x64_nn` speed (5.1 TFLOPS). That is full tuned-kernel or
flash-kernel work, which is card-13 scope and does not fit the time box.

Decision: stop. Branch night15-prefill28b was not created, prod libs are unchanged and no N3 row was run.
