# KQ1 result: q8_0 K lane (staged q8 decode + q8 GEMM prefill)

| gate (fixed) | value | verdict |
|---|---|---|
| engagement: occamy_kq1 stage>0 and occamy_fa_gemm kq8>0 | pending GPU | - |
| backend-ops FLASH_ATTN_EXT eval (stage and plain kvq8) | pending GPU | - |
| decode micro F16/Q8 staged, 1-row and 2-row, 96K (kill <1.05x) | pending GPU | - |
| KLD K vs F16: dmean <= +.0010, top >= -.3 at 32K and 96K | pending GPU | - |
| N3 short, 6 ABBA pairs, makespan gain >=1% at diff/sd_prod >=3 | pending GPU | - |
| prefill_s not worse than +0.2% | pending GPU | - |
| token divergence vs prod | pending GPU (reported, not gated) | - |

No GPU run was done by the coder. Launch `kq1_gpu.sh` under the runner with N3_GPU, N3_UNLOAD_URL, N3_PORT and N3_HOST set in the environment. It takes the GPU lock itself, and summary lines go to `run-*/steps.log`.

## Build
- Branch kq1, base d4c13eb, HEAD d10f8db. Hashes and build command are in RECEIPT.txt.
- Staged kernel `ofa_decode_hs_ar_q8_kernel<KQ8,VQ8,true>`: 128 registers, no local memory (no spills), 42368 B static smem. 2 blocks/SM fit (launch_bounds 256,2).

## Design
- **Decode (opt-in, GGML_CUDA_OCCAMY_KVQ8_STAGE=1, needs OCCAMY_KVQ8=1).** Each warp double-buffers its K tile (q8_0 blocks) into shared memory and syncs with a named barrier per warp, so the global K load for tile i+1 overlaps the dot products of tile i. Rows that are misaligned fall back to the plain kvq8 kernel (counted as stage_misalign).
- **Prefill.** The Maxwell GEMM FA path now accepts a q8_0 K: it is dequantised to f32 (d*qs) in convert_kv. V stays F16.
- **Counters.** At exit: `occamy_kq1: stage= kvq8_plain= stage_misalign= stock_kq8=` and `occamy_fa_gemm: ... kq8=`.

## Harness
- `kq1_gpu.sh` runs these steps in order:
  - (a) backend-ops eval
  - (b) decode micro with three env sets, written to b_micro.md; a kill here skips c and d
  - (c) KLD arms F, K and Kg on docs 0-3, at buckets 32K and 96K
  - (d) N3 short ABBA, 6 pairs of prod-libs batch8 vs KQ1 build
  - (e) counters
- `kq1_n3_pair.py` computes the paired gates. `kld/kq1_kld_analyze.py` computes the KLD gate. `n3_replay_kq1.py` is the replay driver: arm kq1 runs batch8 env + KVQ8 + STAGE with `-ctk q8_0 -ctv f16` and q8 slots.

## Risks
- KLD margin is thin. Q8S measured +.00216 dmean at 32K for stock q8 K, which fails the gate. The Kg arm will show whether that cost comes from the cache type and not from the kernel.
- Mixed batches with fewer than 64 query rows on a q8 K still use stock FA (counted as stock_kq8).
- The decode gain depends on staging hiding latency at 128 registers. This has not been measured.
