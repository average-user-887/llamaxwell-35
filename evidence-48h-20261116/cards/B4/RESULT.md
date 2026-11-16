# B4 multirow decode FA - GPU results (run: <W>/B4/run-20261115T220204Z)

| k | cost vs k=1 at 90K, 2 seqs (auto R / best R) | accept | stock k>1 fallback | speedup vs stock | RMSE vs FP64 stock -> multirow (best/auto R) |
|---|---|---|---|---|---|
| 1 | 1.349 ms (1.00x) | - | same path | - | 2.5e-9 (ref) |
| 2 | 2.041 ms, 1.51x (R2) | - | 3.398 ms | 1.66x | not measured |
| 3 | 4.513 ms 3.35x auto R4; best R1 2.991 ms 2.22x | <=1.15x FAIL | crashed (illegal mem access) | n/a | 5.25e-9 -> 4.96e-9 (R0/4), 4.30e-9 (R1): PASS |
| 4 | 3.735 ms, 2.77x (R0/4) | - | crashed | n/a | not measured |
| 8 | 7.518 ms, 5.57x auto R4; R2 8.16, R1 10.38 | <=1.5x FAIL | crashed | n/a | 7.25e-9 -> 6.95e-9 (R0/2/4), 6.06e-9 (R1): PASS |
| 16 | 15.07 ms, 11.2x auto R4; R1/R2 ~21 ms | - | crashed | n/a | 9.90e-9/1.01e-8 -> 9.98e-9 (R0/4), 8.6e-9 (R1/2): PASS (within noise, ~+1% R4 stream0) |

- RMSE gate (no worse than stock): PASS at k=3,8 all R; k=16 R0/R4 stream0 9.975e-9 vs stock 9.903e-9 (+0.7%, stream1 lower), R1/R2 better.
- test-backend-ops eval vs CPU (multirow on): FAIL 6/12. kv=4096 all OK (k 1,2,3,5,8,16, incl. engaged k=8). All 6 kv=90112 cases FAIL, ERR ~0.0019 > 0.0005, INCLUDING nb=1 (not a multirow path), so likely f16/CPU-reference tolerance at 90K, not multirow specific; fa_rmse14 vs FP64 is clean. Not resolved.
- Engagement counter line seen: yes ("occamy FA multirow on: engaged=N k=.. R=.. ns=2 n_kv=90112 nsplit=..").
- Stock perf run (perf_stock) crashed with CUDA illegal memory access at k=3 after k=1,2 (stock k>1 fallback bug or test-case issue; rc=139), so stock cost for k>=3 is missing.
- Auto R (provisional) is poor: k=3 auto R4 4.51 ms vs R1 2.99 ms; k=8 R2/R4 ok but R1 worst. Suggest auto R: k=3->1, k=2->2. R0 repeat reproduces R0 within 0.2%.
- Roofline note below still applies; ratio cards (k=3<=1.15x, k=8<=1.5x) not met.

Files: <W>/B4/run-20261115T220204Z/{steps.log,rmse/,tbo/}. Script copy with flock -w 43200: <W>/B4/b4_gpu_wait.sh (only change vs b4_gpu.sh).

---
(original pre-run notes follow)

# B4 multirow decode FA (k rows per stream) - code done, GPU numbers pending

| item | value |
|---|---|
| commit | b4-multirow c9e417a (base bundle9 52c9e84) |
| libggml-cuda.so sha256 | d0f32824...d33928 (full list in RECEIPT.txt) |
| cost k=3 / k=1 at 90K, 2 seqs | pending GPU (accept <= 1.15x) |
| cost k=8 / k=1 | pending GPU (accept <= 1.5x; roofline says not reachable, see below) |
| RMSE vs FP64 | pending GPU (k=1 reference 1.3e-4 card 14) |
| regs / spills | mr<1,2,4>: 116/116/117, 0 spill (hs_ar 118) |
| k=1 | unchanged code path (ofa_decode_hs_ar_kernel / ragged), bit-identical by construction |

GPU microbench not run: bench lock held by R0 ABBA (gpu_r0.sh) with B1 queued behind it; the 900 s wait expired.
Ready to run: `N3_GPU=<gpu> N3_UNLOAD_URL=<lab-endpoint>/api/models/unload/qwen36-35b-moe B4/b4_gpu.sh` (~5 min hold).
It records fa_rmse14 (kv 90112, NS 2, k 1/3/8/16, stock vs multirow R auto/1/2/4), test-backend-ops eval vs CPU, and perf at 90K for k 1,2,3,4,8,16.

## Method
- Gate GGML_CUDA_OFA_MULTIROW=1 (needs FA=1, PIPE=3, ADDR=1, F16 K/V). Engages for Q->ne[1] in 2..16; mask must have >= k rows.
- ofa_decode_mr_kernel<R>: 256 threads = 4/R unit lanes x 2R query groups (R rows x 2 head groups of 4 q-heads); each warp runs the hs_ar body
  with Q row r and mask row r (Mb + r*nbm1). K/V of a unit are fetched once per block, the other row warps re-read them from L1/tex;
  row blocks of one split are adjacent in grid x so with R < k they share K/V through L2. Rows past k in the last row block skip the loop.
  ofa_combine_mr_kernel writes dst + i3*nb3 + r*nb2 + qh*nb1. nsplit = ceil(resident/(nkvh*ns*nrb)). R override GGML_CUDA_OFA_MR_R=1|2|4;
  provisional auto R = 4 for k%4==0 or k==3, else 2 (set from the perf run).
- Counter: "ggml_cuda: occamy FA multirow on: engaged=N k R ns n_kv nsplit nkvh" on power-of-two calls.
- test-backend-ops: test_flash_attn_ext_mr_causal (2 streams, causal in block, stream gap 777), eval kv 4096/90112 x k 1,2,3,5,8,16; perf kv 90112 x k 1,2,3,4,8,16.

## Risks / roofline
- FP32 math only on sm_52 (no fast FP16). Per row per layer at 90K, 2 seqs: 1.47e9 FMA = 0.43 ms at peak; current body ~64% FMA share -> ~0.68 ms/row.
  KV read at k=1 ~368 MB -> ~1.2 ms. So k=3 ~2.0 ms (~1.5x) and k=8 ~5.4 ms (~4x) with this body; <=1.15x at k=3 needs near-peak FMA with perfect overlap,
  <=1.5x at k=8 is below the FP32 compute floor (~2.6x). Expect a large gain over the stock k>1 fallback, not the card ratios.
- Ragged two-slot split not used for k>1 (shorter stream streams masked tail); B5 can add it. kvq8 path still k=1 only.
- Next lever if ratios matter: GEMM-style tile (K/V tile in smem, scores without per-score warp reductions), which also suits B6.
