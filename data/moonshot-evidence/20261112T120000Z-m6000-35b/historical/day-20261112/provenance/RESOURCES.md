# Kernel resource table (sm_52, Quadro M6000)

Data: `resources.csv` (101 kernels x 4 builds = 404 rows). Columns: build, kernel, regs, static_smem, dyn_smem, threads, blk_per_sm, grid, spill_stores, spill_loads, stack, dispatch, status.

## Provenance of values
| Column | Source |
|---|---|
| regs, static_smem, spills, stack | compiler (`ptxas -v`, sm_52, CUDA 12.8) |
| threads, dyn_smem, grid | source-derived (`__launch_bounds__`, launch code); most grids are runtime-dependent |
| blk_per_sm | calculated from compiler regs+smem and sm_52 limits (65536 regs, 96 KB smem, 64 warps, 24 SMs); NOT measured |
| measured | only GDN REPACK `gdn_fused<0,1,1,0,1>`: 62 regs, 3 blk/SM, grid 72, 0 spills (ledger rows GDN3-B/GDN4); baseline GDN grid 48 at 2 blk/SM |

## Build columns
- `a_C01_71d1e43`: full ptxas log of an ancestor of c9825b1 (predates QKREUSE/PCSHARE/DOWNH kernels; those rows "not available").
- `b_BUNDLE2_b3c8c71`: incremental log, 50 kernels.
- `c_GDN4_0b2d6f4`: incremental log, 28 kernels (dirty tree, merged as 85dbeed).
- `d_B3`: full rebuild log (421f2fe) overlaid with incremental log of final 2436f75 (gdn_fused2 / N2GDN); 96 kernels.

## Key numbers (d_B3)
| Kernel | regs | blk/SM (calc) | spills |
|---|---|---|---|
| moe_v2<0,*> | 87-104 | 2 (41728 B dyn smem, launch_bounds 256,2) | 0 |
| gdn_fused<0,1,1,0,0> | 72 | 2 | 0 |
| gdn_fused<0,1,1,0,1> (REPACK) | 62 | 3 (measured) | 0 |
| gdn_fused2 | 80 | 3 | 0 |
| ofa_decode_kernel<0/1/2> | 200/255/228 | static smem 33664 B | 0 |
| ofa_decode_hs_kernel<true> (PIPE4) | 128 | - | 28 B st/ld, 16 B stack (only kernel with spills) |
| ofa_decode_hs_kernel<false> | 124 | - | 0 |
| ofa_decode_hs_ar_kernel (FA_ADDR) | 118 | - | 0 |
| head_rowpack_sm52 | 29 | 16 | 0 |

## Gaps
- No full ptxas log exists for baseline c9825b1; column a is the nearest full log (older commit).
- Columns b, c are incremental: unchanged kernels read "not available (not in this build log)".
- B3 `build.log` is incremental; the full rebuild is `build-421f2fe.log` (421f2fe -> 2436f75 differs only by N2GDN).
- Occupancy is calculated, not profiled. Grids depend on runtime shapes.
- Excluded families: dequantize_*, get/set_rows, quantize_mmq_*, mmq tile helpers (prefill), upstream gated_delta_net_cuda fallback, nvfp4/mxfp4 and other quant types, upstream flash_attn_* (unused in decode), topk_moe for non-256 experts.
