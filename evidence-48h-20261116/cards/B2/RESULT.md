# Card B2 RESULT: per-step component timing, k=1 vs k=2 verify rows

Mean ms per decode step (one target verify call plus its draft calls), from the LD_PRELOAD launch profiler. Data: `run-20261116T044254Z` (the valid run). k=1 is n-max 1 (2 verify rows), k=2 is n-max 2 (3 rows). k=0 (spec off) is the reference.

| case | metric | k=0 | k=1 | k=2 | k2/k1 |
|---|---|---|---|---|---|
| 80K, 1 slot | attention | 6.0 | 16.3 | 22.4 | 1.37 |
| | MoE | 4.3 | 8.0 | 8.4 | 1.05 |
| | GDN | 3.9 | 1.7 | 2.2 | 1.29 |
| | dense (mmvq etc.) | 1.0 | 6.2 | 9.9 | 1.60 |
| | other/norm/copy | 1.8 | 5.6 | 8.2 | 1.46 |
| | target GPU total | 17.1 | 37.8 | 51.0 | 1.35 |
| | MTP head (draft calls) | - | 16.2 | 21.0 | 1.30 |
| | host/sampling gap | 2.3 | 3.7 | 1.2 | - |
| | step wall | 19.4 | 57.6 | 73.1 | 1.27 |
| 90K, 1 slot | attention | 6.0 | 18.3 | 25.1 | 1.37 |
| | target GPU total | 17.7 | 39.5 | 53.8 | 1.36 |
| | MTP head | - | 17.5 | 22.3 | 1.27 |
| | step wall | 18.3 | 59.5 | 78.1 | 1.31 |
| 80K, 2 slots (one step covers both) | attention | - | 37.7 | 49.3 | 1.31 |
| | MoE | - | 11.7 | 17.5 | 1.50 |
| | GDN | - | 2.9 | 3.9 | 1.34 |
| | dense | - | 21.8 | 29.4 | 1.35 |
| | target GPU total | - | 86.9 | 113.6 | 1.31 |
| | MTP head | - | 35.3 | 37.5 | 1.06 |
| | step wall | - | 124.0 | 149.8 | 1.21 |

Step ratio k2/k1 (wall): 1.27 (80K), 1.31 (90K), 1.21 (2 slots 80K). k=0 2-slot reference: about 28.7 ms/step (earlier run, same shim; this run's off_np2 2-slot grouping failed, see caveats).
k=0 attention is the single ofa_decode_hs_ar kernel (10 launches/step, 6.0 ms). 1-slot launches/step: k0 225, k1 945, k2 1735.

## Fallback kernels launched at rows>1 (not present at rows=1), per step, 80K 1 slot
Attention is the main one. Both ofa decode and occamy GDN are replaced by generic kernels.
- `flash_attn_tile<256,256,2,8>` (k=1) / `<256,256,4,8>` (k=2): 10 launches/step, 16.0 ms / 22.1 ms. Replaces ofa_decode_hs_ar (6.0 ms). Largest single cost.
- `mul_mat_vec_q<q4_0,ncols=2|3>`: 189 / 311 launches, 6.1 / 8.7 ms (rows=1 version: 40 launches, 1.0 ms).
- `mul_mat_vec_q_moe<q4_0,2>` (k=2 only): 120 launches, 7.5 ms. k=1 stays on occamy `moe_v3` (79.6 launches, 8.0 ms; k=0 40 launches, 4.3 ms).
- `gated_delta_net_cuda<128,..,4>`: 30 launches, 1.4 / 1.9 ms, replaces `occamy_gdn::gdn_fused` (3.8 ms). Plus `ssm_conv_f32`, `concat_non_cont` (30, 1.1 ms), `k_get_rows_float_vec`, `cpy_scalar` (60 / 100).
- `mul_mat_vec_f<3>` (80, 1.2 ms), `topk_moe_cuda` (40, 0.55 ms), `moe_weighted_reduction_f32` (40), `quantize_q8_1` (41 / 241, 0.3 / 1.2 ms), `rms_norm_f32<1024>` (81), silu/sigmoid/softplus unary kernels (30-110).
- Totals: 30 new kernel types at k=1 (670 launches, 28.2 ms); 21 at k=2 (1543 launches, 49.6 ms).
- 2 slots add `mul_mat_q<q4_0,64>` (about 2.4-2.8 launches/step, 10.6-11.8 ms; batched prefill-style path) and `mul_mat_vec_q<ncols=4|6>`.
- Server counters: only `occamy_fa_gemm engaged=0` and `occamy_fa_gemm_fused calls=0` exist; no spec-specific fallback counter.

## Accept rate
- Repaired fixture (n2_np2, mtp n-max 2, 3 turns, ctx 36864, run_sha 144461efdc0bd2f7, run `run-20261116T032103Z`): n_void = 4 of 6 (all 3 slot-1 turns are short voids, 2-3 tokens; slot-0 turn 0 is a loop void).
  Non-void turns (2): 584 drafted, 515 accepted, 0.88. All 6 turns: 1123/1050 = 0.935 (inflated by the looping turn).
- Client runs at n-max 2: 80K 1 slot 105/169 (0.62, run 2) , 90K 1 slot 121/138 (0.88); 80K 2 slots 109/163 and 108/165 (0.66, run 3).
- n-max 1: 93/97 and 92/98 (0.95), 2 slots 95/96 and 88/102.

## Method
No nsys, nvprof or CUPTI exists on host or in the image, so `kprof.c` (LD_PRELOAD) intercepts cudaLaunchKernel(ExC/Cooperative) and memcpy; one cudaEvent is recorded per launch, kernel time is the per-stream event delta, and a base event is reset after each D2H marker so idle gaps are mostly excluded. A segment ends at each D2H memcpy (one decode call). Target steps are segments with at least 0.6 of the arm's max launch count; the smaller segments in between are the MTP head. Component class by demangled name regex (`analyze.py`). Host gap is the time between consecutive target segment ends minus target and MTP GPU ms. Phase labels (s1_d80000, s1_d90000, s2_d80000) come from the phase file the client flips. Arms: ub1024, batch8 libs, q8 draft KV, ctx 98304 (1 slot) or 2x84224 (2 slots), 80K cold prefill then 192 decode tokens, exact-prefix extension to 90K.

## Caveats
- Per-kernel event deltas include small queueing effects; absolute ms overstate by the shim host overhead. Shim 1-slot k=0 gives 17.1 ms GPU vs 19.4 ms wall; client k=0 56.8 t/s vs 58.4 in M2.
- Classifier is regex-based; "dense" for k>=1 includes mul_mat_vec_q used for expert/projection matmuls. Treat sub-1 ms splits as approximate.
- Run history: run 1 shim crashed (dlsym NULL); run 2 prof.txt was truncated by child processes reopening it with "w" (lost labels and names), so it was rerun as run 3 with per-pid output (file `prof.txt.1` is the server, symlinked as `prof.txt`). Run 3 took 06:04 to 06:55 under the lock (about 51 min), above the 30-40 min estimate. Run 2 took about 57 min. Two lock slots plus run 1 were used in total.
- 2-slot at 90K infeasible (2x98304 does not fit; 2x84224 already peaks at 24.4 GB).
- off_np2 s2_d80000 row in analyze output is a grouping artifact (4 steps); k=0 2-slot reference is taken from run 2 (28.7 ms).
- n-max 3 and the debug arm were dropped from run 3 (n3 had 34 segments in run 2: too sparse). Fixture was run only in run 2 (n2_np2), not repeated.
- Fixture is not fully repaired: slot 1 (tool-call, copy-heavy) still emits EOS after 2-3 tokens.
- Single run per arm, no repeats; step counts 69-99 per cell.

## Files
`<W>/B2/`: `run-20261116T044254Z/<arm>/{prof.txt.1,result.json,client.out,server.log,counters.txt,vram.csv}`, `run-20261116T044254Z/kern_detail.json`, `run-20261116T044254Z/steps.log`, `run-20261116T032103Z/n2_np2/fixture/{turns.md,run.json}`, `kprof.c`, `kprof.so`, `b2.sh`, `b2_client.py`, `analyze.py`.
