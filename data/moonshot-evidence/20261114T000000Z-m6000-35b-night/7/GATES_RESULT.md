# Card 7b stack gates

| check | result |
|---|---|
| 1 FA RMSE vs FP64, fused <= stock at all 3 shapes | FAIL |
| 2a temp-0 tokens, stack (GEMM+FUSED+S1) vs GEMM-only, 3 prompts x 256 tok | FAIL |
| 2b 2-slot concurrent with --prefill-single-stream vs own single-slot output | FAIL |
| 3 slot save/restore round trip, tokens identical | PASS |

## 1 Numerics (ne3=1, d=256, 16q/2kv heads, rows sampled step 16)

| kv | nq | stock RMSE | GEMM RMSE | fused RMSE | fused==stock bits | fused==GEMM bits |
|---|---|---|---|---|---|---|
| 8448 | 512 | 1.3431e-08 | 1.3394e-08 | 1.3432e-08 | False | False |
| 32768 | 512 | 1.2530e-08 | 6.7698e-09 | 6.7215e-09 | False | False |
| 98304 | 256 | 9.9960e-09 | 3.8656e-09 | 3.8964e-09 | False | False |

## 2a Single-slot token identity (256 tokens, temp 0)

| prompt | stack vs gemm-only | stack vs stock | gemm-only vs stock |
|---|---|---|---|
| p1_600 | identical | identical | identical |
| p2_6000 | DIFF@10(len 256/256) | DIFF@10(len 256/256) | DIFF@177(len 256/256) |
| p3_30000 | identical | DIFF@32(len 256/256) | DIFF@32(len 256/256) |

## 2b Concurrent two slots (stack, --prefill-single-stream)

| prompt | vs own single (stack) | vs single (gemm-only) |
|---|---|---|
| p2_6000 | DIFF@237(len 256/256) | DIFF@10(len 256/256) |
| p3_30000 | identical | identical |

## 3 Round trip (stack, slot 0, p2_6000, 64 tokens)

- live vs fresh re-prefill: identical
- fresh vs restored: identical
- save/restore: {"save": {"http": 200, "n_saved": 6063, "bytes": 321933668}, "restore": {"http": 200, "n_restored": 6063}}
- restored request cache_n=5996 prompt_n=4

Logs: <LAB>/occamy35-night-20261114/7/gates-run-20261114T124218Z
