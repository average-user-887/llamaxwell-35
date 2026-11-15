# Card 14: ragged two-slot split (GGML_CUDA_OFA_RAGGED=1): built, GPU run not yet done

| Item | Status |
|---|---|
| Code | commit 6b837ea on night14-ragged14 (base stack 246209f) |
| Build (CPU, sm_52) | ok; ofa_decode_rg_kernel 114 regs, 0 spill, 33.7 KB smem, 2 blocks/SM; hs_ar unchanged (118 regs) |
| Gate harness CPU self-check | ok (fa_rmse14, NS=2, gap+hole, RMSE 1.3e-4 vs FP64 on CPU backend) |
| GPU gate / decode ABBA / N3 | NOT RUN: waiter stopped to give the prod swap 7d the lock (orchestrator note) |

## Method
- Flag off, or ns==1, or KV Q8, or PIPE!=3/ADDR!=1: path unchanged (hs_ar plus combine). Single-slot output should stay bitwise the same; gpu_ab.sh part 0 checks this with an OUTHASH comparison on NS=1.
- ns>=2 with F16 KV: `ofa_len_kernel` takes each stream's length from mask row 0 (last column that is not -inf, rounded up to 4). Then `ofa_decode_rg_kernel` runs on grid (NB=ceil(resident/nkvh), nkvh) over one unit space that concatenates all streams. Last, `ofa_combine_rg_kernel` merges only the blocks that touched each stream. Rows that are skipped are all -inf, so skipping them is exact. For ns>=2 the result is not bitwise the same as hs_ar because the split boundaries move.
- Engagement evidence: stderr `occamy FA ragged requested, active ...` at init; `occamy FA ragged on: engaged=N ns n_kv NB nkvh` at powers of two.

## To run (orchestrator, after the swap)
`N3_GPU=<gpu> N3_UNLOAD_URL=<lab-endpoint>/api/models/unload/qwen36-35b-moe 14/gpu_ab.sh`. It holds the lock, writes `14/run-<ts>/`, and runs these parts:
- part 0: RMSE vs FP64 at kv 8448/32768/98304, NS 1/2, gaps 777/-777/2kv/3 plus a hole, arms stack(7/bin) / 14 off / 14 on, and test-backend-ops FLASH_ATTN_EXT with the flag on;
- part 1: decode ABBA x3, units single_30k/both_30k/single_90k/both_deep/both_mixed (98K+30K); analyze with `run-*/dec/analyze.txt`;
- part 2: N3 3 pairs, stack (7/bin) vs ragged (14/bin); engagement is counted with a recursive grep.
The baseline is always 7/bin.

## Caveats
- Expected gain is mainly in both_mixed (uneven slot depths). both_deep (98K+90K) saves only ~4%; both_30k should be neutral.
- Two-slot token sha may differ from flag-off (reduction order); judge on RMSE vs FP64.

## Files
14/src (branch night14-ragged14), 14/build-n14.sh, 14/bin, 14/gates/fa_rmse14.cpp, 14/gpu_ab.sh, 14/n14_dec.py, 14/n14_dec_analyze.py, 14/n14_replay.py, 14/RECEIPT.txt, 14/SHA256.txt
