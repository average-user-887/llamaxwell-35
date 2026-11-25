# LlamaMaxwell-35 R3 Card 3: Standalone Chunked WY GDN Prefill (`B=64`) Study (Retired)

Evaluated Chunked WY Gated Delta Net (GDN) prefill (`B=64`, `S_v=128`, `H=32`, `H_k=16`) standalone against Maxwell (`sm_52`) `stock_gdn_kernel` and FP64 reference before any `llama.cpp` integration.

## 1. Compiler & Numerical Verification (`0` spills, `11/11 PASS`)

- Compiled `src/gdn_wy_bench.cu` for `sm_52` with `-Xptxas -v,-warn-spills -lcublas`: `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads` on all kernels (`wy_prepack_kernel`, `wy_invert_l_kernel`, `wy_mask_qgram_kernel`, `wy_chunk_epilogue_kernel`, `wy_unpack_out_kernel`, `stock_gdn_kernel`).
- Computing cumulative decay `g_cum` in `double` inside `wy_prepack_kernel`/`wy_invert_l_kernel`/`wy_mask_qgram_kernel` and deferring `scale` to `wy_unpack_out_kernel` eliminated `G_j - G_r` subtraction cancellation:
  - `11/11` cases passed (`T in {124, 128, 512, 1024, 2048} x K in {1, 2}` plus harsh decay `g in [-5.0, -0.0001]`, `T=512, K=2`).
  - `wy_dst` rel RMSE: `1.71e-07` to `2.12e-07` (`1.56x - 1.93x` stock's `1.06e-07 - 1.10e-07` vs FP64).
  - `wy_st` rel RMSE: `5.31e-08` to `6.40e-08` (`0.53x - 0.64x`, identical to stock state).
  - `0` non-finites across all outputs and states.

## 2. ABBA Timing & Retirement Decision (`K=2`)

| Length `T` | Stock avg | Chunked WY avg (`B=64`) | Speedup | Delta / call | 16K projected (240 calls) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `3.9451 ms` | `5.0413 ms` | **`0.783x`** | `-1.0962 ms` | `-263.1 ms` slower |
| `2048` | `7.8632 ms` | `10.1562 ms` | **`0.774x`** | `-2.2930 ms` | `-550.3 ms` slower |

- **Why Chunked WY loses on Maxwell (`sm_52`)**: Without Tensor Cores, `stock_gdn_kernel` keeps the `[128, 128]` per-head state resident in SM registers across all tokens with minimal DRAM traffic (`~1.5 KiB/token`), whereas Chunked WY loops over `T/64` sequential chunks of global-memory `32 x (128x128)` `cublasSgemmStridedBatched` calls + chunk epilogue launches (~3 launches per 64 tokens).
- **Gate decision**: Speedup `< 1.5x` (`0.774x - 0.783x`), so Card 3 was retired cleanly without altering `combined-r3-card1`.

## Verification
```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
