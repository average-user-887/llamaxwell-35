# LlamaMaxwell-35 R3 Card 1: Verified Engagement, 80K CUDA Graphs, and Counterbalanced Replay

Follow-up verification packet on `combined-r3-card1` (built with `sm_52`-only CUDA slimming, `T1-T3+A2` truncation/UTF-8/compact-draft fixes, D1b 2-slot capture safety guard, `GGML_CUDA_MAXWELL_GRAPHS=0` fix, and exit counters `r3_cuda_graphs` / `r3_engaged_*`).

## 1. DSO Build & Verification Hashes (`0` unexpected register spills, `ldd -r` clean)

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `libggml-cuda.so` | 14,679,848 B | `c08bb3f08af3c0385f438836188845ff535587dad0c4ca63b5310806a65cd59f` |
| `libllama.so.0.6.0` | 5,083,544 B | `85c6322def25b9e9141242e6d5589fc65bae0d24f6536f51a40dace1153ae2a4` |
| `libllama-common.so.0.6.0` | 6,524,120 B | `3ad7d0ff8fe78b31679cfc0d59561a5a74cb5be1d375bdf066a332a01feacab0` |
| `libllama-server-impl.so` | 7,715,464 B | `bdd39cf9a6cf9bfa51271ce5c6978bba3acef9f570214d994a71b3fd9945fe5e` |

Full 64-char SHA-256 values are checked in `results.json`. Patches included in `patches/`:
- `ggml-cuda.cu.patch`: D1b 2-slot capture guard (`ne[3]==2 && ggml_cuda_occamy_n2split()`), `GGML_CUDA_MAXWELL_GRAPHS="0"` handling, `r3_cuda_graphs` counters.
- `occamy-fused.cu.patch`: `r3_engaged_moe` (`persist_scratch2`, `moe_sh_q8`).
- `fattn-occamy-sparse.cuh.patch`: `r3_engaged_fa` (`q1_r2_fast`, `q2_r2_stride`).
- `occamy-gdn.cu.patch`: `r3_engaged_gdn` (`q8c`, `q8a`, `fp32`).

## 2. `t123` + `fast2k` AB/BA Component Isolation (`--unset GGML_CUDA_OCCAMY_DEBUG`)

- `t123` passed all 5 gates (`T1_sf2_rate_limit`, `CardA_and_A2_shadow` `49/49`, `CardA_and_A2_active` `119/119` `0` fallbacks, `T2` `2052/2179`, `T3` UTF-8), with `r3_cuda_graphs: calls=277 arch_off=0 incompat=0 captures=18 replays=202`.
- Counterbalanced `fast2k` AB/BA means:
  - `c1-base`: `116.67 tok/s` (`16.498 ms/r`), `cg arch_off=751`
  - `c1-nograph` (`+A2+PERSIST_SCRATCH2+MOE_SH_Q8+FA_Q1_R2_FAST`): `118.92 tok/s` (`16.248 ms/r`, `-0.250 ms/r`), bit-identical output SHA
  - `c1-graph` (`+GGML_CUDA_MAXWELL_GRAPHS=1`): `119.21 tok/s` (`16.207 ms/r`, `-0.041 ms/r` warm), `cg captures=37 replays=622 incompat=0`, bit-identical
  - `c1-q8c` (`+GGML_CUDA_OCCAMY_GDN_T2_Q8C=1`): `121.32 tok/s` (`15.925 ms/r`, pure delta `-0.282 ms/r` warm at identical `245/265` acceptance)

## 3. 80K `dec_profile` Attribution

- `dec2` (`268` graphs, `3891.51 ms` total node time): `RMS_NORM@MoE` `38.01%` (`5.519 ms/g`), `RMS_NORM@GDN` `28.59%` (`4.152 ms/g`), `FLASH_ATTN_EXT` `11.30%` (`1.640 ms/g`), `MUL_MAT:output.weight` `8.87%` (`1.288 ms/g`).
- `mv1_host`: `draft_ms=510.930` over `237` rounds (`2.156 ms/round`).
- Diagnosed `ctx3 n=2` `prep_us = 153652.0 us` across `30` calls: `server.log` confirmed `151.93 ms` came from one-time startup `sched_reserve` during the `load_model` warmup probe (`~59 us` per reject across the 29 real rejects).

## 4. Counterbalanced 2-Session 80K Verification

| Suite | Arm | Avg TTFT | Avg Cold | Avg Warm | Warm ms/r / Parity |
| --- | --- | ---: | ---: | ---: | --- |
| `warm80k` | `r2` baseline | `265.93 s` | `101.91 tok/s` | `102.15 tok/s` | `18.314 ms/r`, ref |
| `warm80k` | `exact` (`R3-lossless`, `ub=1024`) | `265.93 s` | `102.73 tok/s` | `104.71 tok/s` | `17.879 ms/r`, `cold_ex=True, warm_ex=True` |
| `warm80k` | `ub2048v5` (`-ub 2048 --mixed-batch 1024 + V5`) | `252.50 s` | `103.12 tok/s` | `105.11 tok/s` | `17.876 ms/r`, diff at tok `45` |
| `warm80k` | `q8c` (`+GDN_T2_Q8C=1`) | `265.54 s` | `106.43 tok/s` | `108.79 tok/s` | `17.660 ms/r`, `q8c=930`, diff at tok `27` |
| `warm80k` | `serial` (`--mode serial`) | `261.53 s` | `81.10 tok/s` | `80.88 tok/s` | `cg rep=3232/3345` |
| `ordinary80` | `r2` baseline | `264.55 s` | `94.93 tok/s` | `-` | `18.343 ms/r`, ref |
| `ordinary80` | `exact` (`R3-lossless`, `ub=1024`) | `265.28 s` | `96.82 tok/s` | `-` | `17.940 ms/r`, `exact_r2=True` |
| `ordinary80` | `ub2048v5` (`-ub 2048 --mixed-batch 1024 + V5`) | `251.26 s` | `98.40 tok/s` | `-` | `17.937 ms/r`, `exact_r2=False` |

At 80K context (`warm80k exact`), `r3_cuda_graphs` replayed `3045/3651` calls (`captures=195, incompat=0`), `compact_only` reached `1743/1743` (`fb=0` on draft), and all corrected env vars showed positive engagement (`persist_scratch2=1239`, `moe_sh_q8=486`, `q1_r2_fast=496`, `q8a=930`).

## Verification
```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
