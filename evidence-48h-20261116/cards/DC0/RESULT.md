# DC0 decode scaling + launch census - step 1 done (CPU, from L2 data); step 2 built, GPU run NOT executed

## Step 1: L2 run-20261116T085837Z, 1-row decode (748 graphs, 13.64 s timed = 18.2 ms/step, matches J0 18.3 ms)
| class | 1-row ms/step | nodes/step | us/node | 2-row ms/step | ratio 2/1 |
|---|---|---|---|---|---|
| norm/rope (RMS_NORM-led, see caveat) | 8.56 | 121 | 70.7 | not separable | - |
| attention (FLASH_ATTN_EXT, 10 layers) | 6.91 | 10 | 690.7 | not separable | - |
| dense mmq (attn q/k/v/out + lm head) | 2.16 | 41 | 52.6 | not separable | - |
| other (add/cont/view/set_rows...) | 0.62 | 214 | 2.9 | not separable | - |
| GDN, MoE experts | 0.00 | 0 | - | - | - |

L2 cannot give the 2-row column: phase "small" lumps every graph with 2..1023 rows (1196 graphs = ~1149 2-row decode + ~47 mixed/prefill-chunk graphs); its 111.5 s timed (93 ms/graph) is dominated by those big graphs (FA 66.6 s), so 2-row cost cannot be backed out.

Findings from the 1-row data:
- GDN and MoE (MUL_MAT_ID, shexp, router) nodes never appear in decode: they sit in fused chains whose time is booked on the first node (RMS_NORM). So "norm/rope 46.9%" is really norm + fused GDN/MoE layer bodies (8.56 ms/step); the 70 us/node figure is a fusion artefact, not norm cost.
- Attention 0.69 ms per full-attn layer call is KV streaming, not weight-bound (269 GB/s = 85% of BW peak); if the kernel re-reads KV per row it scales ~2x, a prime suspect for the 1.76x (unproven until the 2-row column exists).
- Dense mmq 52 us/node (41 nodes) is far above an 8-12 us launch floor: weight-bound.

## Unattributed ~11 s of N3 wall (host side, not GPU)
J0 idle windows (gap >= 2 s): after a 1-row decode 9.2 / 9.1 / 8.8 s (L_1/L_2/L_3) = slot checkpoint restore/save wait (L2 run restore_s 9.98 s, max_gap 5.6 s: the 5-7 s ckpt wait) plus the remainder to ~9 s; plus 2.0 s after each 2048-token prefill batch (L_2, L_3). Together ~11 s per replay (9 s ckpt/restore gap + 2 s post-prefill gap); nothing else idle is visible in the ledger.

## Step 2 (built; run pending)
Branch dc0 (worktree DC0/src) off l2-nodetime: GGML_CUDA_NODE_TIMING_SPLIT=1 adds phase dec2 (exactly 2 rows), tags norm-led nodes with what was fused after them (@MoE, @GDN, @FA, @fused), counts skipped (fused) nodes, adds per-phase graph ms. Default off. Summarizer dc0_summarize.py prints 1-row vs 2-row ms/step, nodes/step, us/node, ratio, and flags ratio>1.3 or us/node in 8-12.
Run: N3_GPU=.. N3_UNLOAD_URL=.. N3_PORT=.. N3_HOST=.. bash DC0/dc0_gpu.sh (one replay, 3300 s timeout, shared lock, POST-only unload). Output DC0/run-*/summary.md.
Caveat: "launches" = graph nodes plus fused-skipped nodes, not CUDA kernel launches (no nsys/ncu).
