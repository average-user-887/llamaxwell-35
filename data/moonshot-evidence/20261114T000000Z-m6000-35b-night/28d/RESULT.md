# Card 28d (profiling only, no code changes)

## Row 1: achieved fraction of bound, per 2048-token ubatch (card 28 trace, CPU analysis)
| op | depth | work | GPU ms | achieved | % of bound | bound type |
|---|---|---|---|---|---|---|
| MoE mmq (routed experts) | 30K | 4.12e12 FLOP (2048 tok x 8 exp x 3 mat x 2x2048x512 x 40 layers) | 1400 | 2.95 TFLOPS | 43% of 6.9 TFLOPS FP32 | compute (int8 mmq, no dp4a); weights 18.1 GB read = 12.9 GB/s = 4% of 317 GB/s, so not memory |
| MoE mmq | 92K | same | 1414 | 2.92 TFLOPS | 42% | compute |
| dense proj (dequant + SGEMM) | 30K | 5.86e12 FLOP (analytic: attn q/k/v/o x10, GDN qkv/z/out/a/b x30, shared expert x40, router x40) | 1215 (73 dequant) | 4.82 TFLOPS (SGEMM only 5.13) | 70% (74%) | compute FP32 |
| dense proj | 92K | same | 1224 | 4.78 TFLOPS (5.09) | 69% (74%) | compute FP32 |
| dequant fp32 write | both | 5.7 GB | 73 | 78 GB/s | 25% of 317 | memory, only 6% of dense time |

Dims: n_embd 2048, expert ff 512, 256 experts / 8 used, 40 layers (10 full-attn, 30 GDN), Q4_0.
Dense FLOP is an analytic estimate of which matmuls run through dequant+SGEMM (grid dims not decisive); +-10%.

## Verdict
- MoE mmq: HEADROOM. 42-43% of FP32 peak, flat in depth. Ceiling ~2x if an int path reached ~75-80%; needs a kernel (out of scope here). Share 13% at 92K, 25% at 30K.
- Dense proj: little headroom (~70-74%, near the ~5.0 TFLOPS cuBLAS ceiling on sm_52 found in card 28). Possible gain only by skipping dequant (6%).

## Row 2: two-slot decode profile at ~90K: NOT COMPLETE at hand-back time
Script $N2/28d/run28d.sh running (nohup, lock held, container n28d-dec, plain arm then nvprof arm; both slots restore 90000-token slot file, 256 tokens each).
Outputs will appear in $N2/28d/run-*/ (decode-plain.json, decode-prof.json, trace.csv, steps.log). Classify trace with $N2/28/an28.py (GDN-launch segmentation also fits decode steps; check --names for mmvq MoE vs dense by grid z).

## Row 2 (completed): two-slot decode at ~90K, batch8 libs, prod env/args
Both slots restored to 90000 tokens, 256 tokens decoded each, concurrently (257 steps, 2 tokens/step). Evidence: run-20261115T162909Z (plain arm), run-20261115T163429Z (nvprof arm; trace.csv, decode-prof.json).
| arm | tok/s per slot | combined |
|---|---|---|
| unprofiled | 32.4 / 32.2 (events 20..end), server 32.1 / ? | ~64 |
| nvprof | 32.2 / 31.9 | ~64 |
nvprof adds no measurable slowdown here (decode kernels are long, GPU-bound).

Per-kernel share of decode GPU time (about 7.6 s over 257 steps = ~29.6 ms/step, ~95% of the 31 ms wall step):
| class | kernel | ms | share |
|---|---|---|---|
| attention | ofa_decode_hs_ar_kernel (+combine) | 3373 | ~44% |
| MoE | occamy::moe_v3 | 2035 | ~27% |
| GDN | gdn_fused2 (+small) | 1503 | ~20% |
| dense proj | mul_mat_vec_q (int=2, incl. shared expert, head) + quantize | ~585 | ~7.7% |
| norms | rms_norm* | ~47 | ~0.6% |
| other (rope/KV/bcast/gather) | | ~90 | ~1.2% |

## Verdict per op (decode, 90K, two slots)
Bound estimates are rough (not computed from bytes in this card): attention reads KV 2 slots x 90K x 10 layers; no per-kernel byte count was measured, so headroom % of 317 GB/s for decode is NOT stated.
- attention: largest share (44%), memory-bound by KV reads; check bytes/step before claiming headroom (follow-up).
- MoE: 27%; decode reads only the active experts; not a % of peak here.
- GDN: 20%, state-read bound; candidate for a look.
- dense: 8%; norms/other: ~2%, no headroom worth chasing.
