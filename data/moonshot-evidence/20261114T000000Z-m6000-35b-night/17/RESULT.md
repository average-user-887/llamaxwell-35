# Card 17: moe_v3 KLD gate (stack+moe3 d4c13eb vs stack 7/bin)

## b98304 (S = moe3 + GGML_CUDA_MOE_V3=1; P = ref stack; floor = Pp = ref + FA_PARALLEL_BLOCKS_MULT=2)
| bucket | S-vs-P dmean | CIdoc | CIrow | floor dmean | floor CIdoc | S top (pt) | floor top (pt) | rows/docs | dmean<=floor | top>=-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 98304 | +0.00000 | [+0.00000,+0.00000] | [+0.00000,+0.00000] | +0.00304 | [+0.00070,+0.00538] | +0.00 | -1.95 | 512/8 | PASS | PASS |

## b32768 (S = moe3 + GGML_CUDA_MOE_V3=1; P = ref stack; floor = Pp = ref + FA_PARALLEL_BLOCKS_MULT=2)
| bucket | S-vs-P dmean | CIdoc | CIrow | floor dmean | floor CIdoc | S top (pt) | floor top (pt) | rows/docs | dmean<=floor | top>=-0.3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32768 | +0.00000 | [+0.00000,+0.00000] | [+0.00000,+0.00000] | +0.00175 | [-0.00071,+0.00422] | +0.00 | +0.00 | 256/4 | PASS | PASS |

Gate: dmean(S-P) <= same-run floor dmean(Pp-P); top-token delta reported (noise-limited, not a hard gate; 7e analyzer threshold -0.3pt shown in last column).

## Method
Harness copied from 7/kld2 (g1c_rows teacher-forced FP32 logits, 64 rows per bucket, docs per 2-doc chunk). Arms: P = 7/bin lib+libs, GEMM+FUSED; S = 9/bin libs + GGML_CUDA_MOE_V3=1; Pp = P + FA_PARALLEL_BLOCKS_MULT=2. Buckets 98304 (docs 0-7, ctx 98560) and 32768 (docs 0-3, ctx 34000); 65536 skipped. MAXWELL_GRAPHS unset. Containers n17-*. Prod unloaded by POST only.
## Caveats
Env sets are from the 7e harness BASE_ENV; LLAMA_PREFILL_SINGLE_STREAM is server-level and not used by g1c_rows. Top-token delta is noise-limited.
## Files
<LAB>/occamy35-night-20261114/17/run_kld.sh kld_harness.py kld_analyze2.py; <LAB>/occamy35-night-20261114/17/run-20261114T232820Z/steps.log, <LAB>/occamy35-night-20261114/17/run-20261114T232820Z/*/analyze.out, <LAB>/occamy35-night-20261114/17/run-20261114T232820Z/*/{P,S,Pp}/

lock taken 23:28:20
unload rc=0 23:28:20
b98304 arm P rc=0 00:18:53
b98304 arm P engaged: occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  00:18:53
b98304 arm S rc=0 01:09:24
b98304 arm S engaged: occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  01:09:24
b98304 arm Pp rc=0 01:59:21
b98304 arm Pp engaged: occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=1860 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  01:59:21
b98304 analyze rc=0 01:59:45
b32768 arm P rc=0 02:06:09
b32768 arm P engaged: occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  02:06:09
b32768 arm S rc=0 02:11:36
b32768 arm S engaged: occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  02:11:36
b32768 arm Pp rc=0 02:17:00
b32768 arm Pp engaged: occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=backend_free occamy_fa_gemm: engaged=580 q_tiled=0 fallback=1340 reasons=q_lt_64:1280,keys_lt_4096:60 when=exit  02:17:00
b32768 analyze rc=0 02:17:08
done elapsed_min=168 02:17:08
cleanup; releasing lock 02:17:08
