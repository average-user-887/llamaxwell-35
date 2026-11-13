# Stock upstream llama.cpp (b12318) baseline on M6000

Peak VRAM: **22987 MiB**

| unit | prompt_n | cache_n | prompt_s | prompt_tps | pred_n | pred_tps | wall_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| warm_4k | 4096 | 0 | 14.87 | 275.47 | 128 | 44.09 | 17.75 |
| prefill_90k_s0 | 85908 | 4092 | 583.68 | 147.18 | 1 | 0.00 | 583.71 |
| prefill_84k_s1 | 84000 | 0 | 538.55 | 155.97 | 1 | 0.00 | 538.58 |
| append_4k_s0 | 4096 | 90000 | 40.63 | 100.81 | 64 | 27.35 | 42.96 |
| both_deep_s0 | 8 | 94092 | 0.26 | 30.19 | 128 | 19.18 | 7.01 |
| both_deep_s1 | 4 | 84000 | 0.09 | 42.65 | 128 | 18.65 | 6.93 |

2-slot deep concurrent wall: **7.014 s**, sum predicted_tps: **37.83 tok/s**
