# M6000 35B state at 13 Nov 2026 17:00

## Prod (swapped ~16:10 13 Nov)
- Container `llamacpp-m6000` (on-demand via host llama-swap; Exited when idle is normal).
- Config: BUNDLE3 env + `GGML_CUDA_FA_MAXWELL_GEMM=1`, **F16 KV**, `-ub 1024`, `--mixed-batch-size 0` (cap off), `-np 2`.
- Create script: `$M/run-35b-b3gemm-20261113.sh` (backup `.pre-ub1024`). Libs: `$M/prod-libs/b3gemm-20261113/`
  (llama-server sha256 3a91fb64…24c5, libggml-cuda.so aa1ab786…23adc, RECEIPT.txt). Mounted at /candidate (ro),
  `--entrypoint /candidate/llama-server`, `LD_LIBRARY_PATH=/candidate`.
- Source: `$N/WP0/src` branch `gemm-b3` (base 2436f75; 826fae7 cap, 0cd59b1 GEMM v4, 71a0822+2022216 counters,
  406856a+abd5584 KVQ8 flag default off).
- Smoke: 9.9K-token prompt correct, 19.1 s; log `occamy_fa_gemm: engaged=70 fallback=140 reasons=q_lt_64:110,keys_lt_4096:30`.

## Rollback (one command line)
`docker stop llamacpp-m6000; docker rename llamacpp-m6000 llamacpp-m6000-b3gemm-failed; docker rename llamacpp-m6000-bundle3-prev llamacpp-m6000`
Next swap: rename current to `llamacpp-m6000-b3gemm-prev`; never remove containers.

## Numbers (WP0 harness, N3 replay profile; details in WP0/harness/RESULT*.md)
| arm | +4K@90K single | 2-slot makespan vs b3 | peak MiB |
|---|---|---|---|
| b3 (old prod) | 31.25 s | 210.68 s | 23109 |
| gemm F16 ub1024 cap0 (**prod now**) | 23.09 s (-26%) | -15.76% (N3B ref -15.69%, n=3) | 23459 |
| gemm + K-only Q8 | 30.47 s, engaged=0 (`ktype`) | -0.46% | 22619 |
| gemm F16 ub2048 cap1024 | 30.4 s, engaged=0 (`q_gt_1024`) | -12.35% (CI -14.36..-10.34) | 23867 |

## Not adopted (owner had accepted these; data says no)
- K-only Q8 KV: GEMM rejects q8_0 K -> card WP3b.
- ub2048: GEMM guard q > 1024 rejects whole ubatch -> card G1 (guard raise).
- cap1024 with GEMM: untested -> card C1.

## Open evidence for the night
- Two-slot GEMM fallbacks: `ne3` 11,710 (both slots in one batch -> GEMM off) -> WP1 is needed.
- `q_lt_64` 7,550 = decode, expected (GEMM is prefill only).
