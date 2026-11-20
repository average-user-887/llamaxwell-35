# LlamaMaxwell-35 R3: Maxwell sm_52 Slim Build, Truncation Fixes, and Decode/Prefill Optimizations

Targeted Phase 2 source patches, CPU unit tests, and single-owner M6000 (`sm_52`, 24 GB) benchmarks for **Occamy 35B-A3B (Qwen 3.6 MoE + MTP1)** on top of the R2 native compact-readback baseline (`101.33 tok/s` warm 80K, `93.71 tok/s` ordinary 80K).

## 1. Truncation & Functional Fixes (All Verified in `gate6`)

1. **T1 (`occamy-sf2` log flood)**: Rate-limited log prints via power-of-two gating -> `22` log lines over `128` serial tokens with debug ON.
2. **T2 (2050-token MTP sub-batch overflow)**: Chunked `dq_emit()` and prefill `common_batch` loops at `llama_n_batch(ctx_dft)` (`2048`), validated live in `gate6` (`turn2_2048_suffix` `cache_n=111, prompt_n=2052`; `turn2_2064_suffix` `prompt_n=2179`).
3. **T3 (Mid-stream + end-of-stream SSE UTF-8 truncation)**: Fixed `incomplete` UTF-8 token emission in `server-context.cpp` so mid-stream buffering and final slot flush never drop multi-byte characters (`split_index_k=2`, `n_out=2, predicted_n=2, valid_utf8=true`).
4. **Sub-card A2 (1-row draft compact argmax + LastQuery restore)**: Enabled 16-byte device-to-host compact argmax readback on `ctx_dft` (`LLAMA_OCCAMY_COMPACT_DRAFT=1`) alongside rebased `LLAMA_MTP_LAST_QUERY=1` (`llama_occamy_compact_draft_last_query`), reaching `119/119` (`0` fallback) on `gate6` with `mtp_last_graph type=5` and `logit=compact`.

- **`gate6` (`t123` on `combined-r3-pp` with `V5=1`)**: `T1=True`, `CardA+A2_shadow=True` (`0` mismatches), `CardA+A2_active=True` (`119/119` target + `119/119` draft compact_only, `exact_parity_vs_shadow=true`), `T2=True`, `T3=True`.

## 2. Maxwell (`sm_52`) Slim DSO & Build Audit

| Artifact | R2 Baseline | R3 Combined (`sm_52`-only) | SHA-256 prefix | Notes |
| --- | ---: | ---: | --- | --- |
| `libggml-cuda.so` | 38,366,536 B | 14,678,296 B (-61.7%) | `824dc96f8a0c` | Occamy+sm_52 only, 0 register spills, `ldd -r` clean |
| `libllama.so.0.6.0` | - | 5,083,544 B | `85c6322def25` | Exports `llama_occamy_compact_draft_last_query` + `valid_row` fix |
| `libllama-common.so.0.6.0` | - | 6,524,120 B | `3ad7d0ff8fe7` | `night-one-graph-draft` rebase + T2 sub-batch + A2 |
| `libllama-server-impl.so` | - | 7,715,464 B | `bdd39cf9a6cf` | T3 UTF-8 mid-stream + final flush fix |

All three CPU test suites compiled cleanly with `-Werror` and passed: `cpu-r3-fixes-test`, `cpu-mtp-subbatch-test` (including `test_last_query_compact_draft`), and `cpu-compact-policy-test`.

## 3. `warm80k` 80K Context Benchmark (`-c 84224`, R2 native ref = `101.33 tok/s`)

| Run | Config (`ub`, flags) | Cold 80K TTFT (tok/s) | Warm Decode (tok/s) | ms/round | Acc (`n_acc/n_dft`) | Parity vs R2 (`cold / warm`) | Uplift vs R2 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `r3_base_lq` | `ub=1024`, new flags OFF | 265.12 s (301.75) | **102.13** | 18.318 | 1429/1642 | exact / exact | +0.80% |
| `r3_exact_lq` | `ub=1024`, lossless TG (`A2+D1+D2+D3+GRAPHS+DBG0`) | 265.72 s (301.07) | **102.91** | 18.181 | 1429/1642 | exact / exact | +1.56% |
| `r3_full_ub2048` | `ub=2048 --mixed-batch 1024`, lossless TG + `V5=1` | 251.79 s (317.73) | **103.56** | 18.165 | 1437/1633 | diff@45 / diff@45 | +2.21% |
| `r3_q8c_lq` | `ub=1024`, `r3_exact_lq` + `GDN_T2_Q8C=1` | 265.73 s (301.06) | **107.28** | 17.931 | 1473/1597 | diff@27 / diff@27 | +5.87% |

## 4. `ordinary80` Unscripted Reading Gate (`-c 84224`, R2 ref = `93.71 tok/s`, 700 tokens)

| Run | Config | 80K TTFT | Decode (tok/s) | ms/round | `n_out` (`finish`) | Parity vs R2 | Uplift vs R2 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `ord80_exact_ub1024_v4` | `ub=1024`, lossless TG (`V5` OFF) | 263.80 s | **95.81** | 18.175 | 700 (word) | exact | +2.24% |
| `ord80_exact_ub1024_v5` | `ub=1024`, lossless TG + `V5=1` | 263.30 s | **94.53** | 18.214 | 625 (word) | diff@19 | +0.87% |
| `ord80_full_ub2048_v5` | `ub=2048 --mixed-batch 1024`, lossless TG + `V5=1` | 251.32 s | **97.53** | 18.192 | 676 (word) | diff@10 | +4.08% |
| `ord80_q8c_ub1024` | `ub=1024`, lossless TG + `Q8C=1` | 264.46 s | **97.68** | 17.960 | 600 (word) | diff@24 | +4.24% |

## 5. `pp_profile` 16K Prefill Microbenchmark (`-c 20480`)

| Run | `ub` / `--mixed-batch` | Flags | Prompt ms | PP tok/s | `prefill` GPU ms | `small` GPU ms | `FLASH_ATTN_EXT` ms | `MUL_MAT_ID` ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ub1024` | `1024 / 1024` | V4 baseline | 34628.41 | **473.14** | 31625.098 | 2432.882 | 5739.87 | 12298.76 |
| `ub2048` | `2048 / 1024` | V4 baseline | 32295.82 | **507.31** | 31586.328 | 59.828 | 6579.15 | 11164.70 |
| `ub2048_v5` | `2048 / 1024` | `GEMM_V5=1` (`MIN_K=4096`) | 31888.13 | **513.80** | 30972.669 | 59.795 | 6103.23 | 11096.56 |
| `ub2048_v5_mink2048` | `2048 / 1024` | `GEMM_V5=1, MIN_K=2048` | 32039.32 | **511.37** | 31145.248 | 61.631 | 6091.79 | 11180.27 |
| `ub2048_v5_mmid` | `2048 / 1024` | `GEMM_V5=1, MMID_CUBLAS_MIN_M=1024` | 52497.89 | **312.09** | 51597.627 | 72.353 | 6114.55 | 31644.04 |

Findings from `pp_profile`:
- `ub=2048 --mixed-batch 1024` collapses GDN/draft small-batch graph overhead (`2,432.88 ms -> 59.83 ms`).
- **`GGML_CUDA_FA_MAXWELL_GEMM_V5=1`** adds `half2->float2` KV conversion, single-group query packing (raising chunk size `1024 -> 2048` with zero extra scratch VRAM -> `q_tiled=0`), and GPU mask-bound scanning (`maxwell_gemm_scan_mask_bounds`) to skip causally masked `-inf` key tiles, cutting 16K `FLASH_ATTN_EXT` from `6,579.15 ms` to `6,103.23 ms` (`-475.9 ms`) and 80K cold prefill from `263.8-265.7 s` down to **`251.3-251.8 s`** (`318 tok/s`).
- `MMID_CUBLAS_MIN_M=1024` regressed `MUL_MAT_ID` by 2.85x (`11,096.6 ms -> 31,644.0 ms`), confirming custom `sm_52` XMAD MMQ strongly outperforms cuBLAS + CPU sort and keeping `MIN_M` OFF.

## 6. `fast2k` Component Ablation (`-c 8192`, `ub=1024`, pre-LastQuery)

| Run | Warm tok/s | ms/round | Acc | Parity vs `base` |
| --- | ---: | ---: | ---: | --- |
| `base` | 116.30 | 16.551 | 244/266 | exact |
| `a2` | 117.12 | 16.434 | 244/266 | exact |
| `d1` | 116.91 | 16.465 | 244/266 | exact |
| `d2` | 116.28 | 16.553 | 244/266 | exact |
| `c_q8c` | 118.01 | 16.372 | 245/265 | diff@38 |
| `stack_exact` | 117.77 | 16.344 | 244/266 | exact |
| `stack_q8c` | 119.83 | 16.123 | 245/265 | diff@38 |
| `graphs` | 118.08 | 16.301 | 244/266 | exact |

## 7. Recommended Runtime Profiles & Phase 3 Next Frontier

- **Lossless Exact Profile (100% bit-exact vs R2 at `ub=1024`)**: `LLAMA_MTP_LAST_QUERY=1 LLAMA_OCCAMY_COMPACT_DRAFT=1 GGML_CUDA_OCCAMY_GDN_PERSIST_SCRATCH2=1 GGML_CUDA_OCCAMY_MOE_Q2_DOWN4_Q8=1 GGML_CUDA_OCCAMY_FA_Q1_R2_STRIDE=1 GGML_CUDA_MAXWELL_GRAPHS=1 GGML_CUDA_OCCAMY_DEBUG=0` -> **`102.91 tok/s`** (`warm80k`), **`95.81 tok/s`** (`ordinary80`).
- **Full E2E Prefill + Decode Profile (`ub=2048 --mixed-batch 1024`)**: add `GGML_CUDA_FA_MAXWELL_GEMM_V5=1` -> **`251.3-251.8 s`** 80K TTFT (`-12.5` to `-13.9 s`), **`103.56 tok/s`** (`warm80k`), **`97.53 tok/s`** (`ordinary80`); `first_diff` at token 10-45 vs `ub=1024` from FP32 GEMM/softmax reduction order.
- **Opt-in Q8C GDN Profile (`+ GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`)**: **`107.28 tok/s`** (`warm80k`, `+5.87%`), **`97.68 tok/s`** (`ordinary80`, `+4.24%`), `first_diff` at token 24-38; treat as experimental pending full BFCL + IFEval quality gating.
- **Phase 3 targets**: (1) chunked WY / parallel-scan GDN for prefill (`1,907 ms` at 16K), (2) device-to-device copy for `t_h_nextn -> embd_state` avoiding host round-trip, and (3) dense FP32 prefill GEMM tuning (`attn_qkv` `3.30 s`, `ssm_out` `1.75 s`, `attn_gate` `1.66 s`).
