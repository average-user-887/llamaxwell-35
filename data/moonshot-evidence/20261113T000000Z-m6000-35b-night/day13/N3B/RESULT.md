# N3B result (gemm = integration-v4 lib sha 0c7ffbde, b3 = bundle3; short profile, ub 1024, cap 0, 3 blocks)

Runs ok: b0_p0_b3=True, b0_p1_gemm=True, b1_p0_gemm=True, b1_p1_b3=True, b2_p0_b3=True, b2_p1_gemm=True, prof_gemm=True

## prompt_ms per request (agent=slot, turn: kind, prompt_n) by block

| block | arm | slot | turn | kind | prompt_n | prompt_ms | tok/s |
|---|---|---|---|---|---|---|---|
| 0 | b3 | 0 | 0 | 1K | 1024 | 7622.0 | 134.3 |
| 0 | b3 | 0 | 1 | 4K | 4098 | 38412.6 | 106.7 |
| 0 | b3 | 0 | 2 | 1K | 1026 | 8049.0 | 127.5 |
| 0 | b3 | 0 | 3 | 4K | 4098 | 40473.2 | 101.3 |
| 0 | b3 | 1 | 0 | 1K | 1024 | 15512.7 | 66.0 |
| 0 | b3 | 1 | 1 | 4K | 4098 | 30960.0 | 132.4 |
| 0 | b3 | 1 | 2 | 1K | 1026 | 28247.6 | 36.3 |
| 0 | b3 | 1 | 3 | 4K | 4098 | 32547.7 | 125.9 |
| 0 | gemm | 0 | 0 | 1K | 1024 | 5629.7 | 181.9 |
| 0 | gemm | 0 | 1 | 4K | 4098 | 32321.2 | 126.8 |
| 0 | gemm | 0 | 2 | 1K | 1026 | 5961.6 | 172.1 |
| 0 | gemm | 0 | 3 | 4K | 4098 | 34152.9 | 120.0 |
| 0 | gemm | 1 | 0 | 1K | 1024 | 15496.2 | 66.1 |
| 0 | gemm | 1 | 1 | 4K | 4098 | 22915.5 | 178.8 |
| 0 | gemm | 1 | 2 | 1K | 1026 | 23869.7 | 43.0 |
| 0 | gemm | 1 | 3 | 4K | 4098 | 23974.0 | 170.9 |
| 1 | gemm | 0 | 0 | 1K | 1024 | 5617.8 | 182.3 |
| 1 | gemm | 0 | 1 | 4K | 4098 | 32302.8 | 126.9 |
| 1 | gemm | 0 | 2 | 1K | 1026 | 5946.4 | 172.5 |
| 1 | gemm | 0 | 3 | 4K | 4098 | 34062.9 | 120.3 |
| 1 | gemm | 1 | 0 | 1K | 1024 | 15488.6 | 66.1 |
| 1 | gemm | 1 | 1 | 4K | 4098 | 22898.3 | 179.0 |
| 1 | gemm | 1 | 2 | 1K | 1026 | 23798.4 | 43.1 |
| 1 | gemm | 1 | 3 | 4K | 4098 | 23950.1 | 171.1 |
| 1 | b3 | 0 | 0 | 1K | 1024 | 7568.5 | 135.3 |
| 1 | b3 | 0 | 1 | 4K | 4098 | 38276.7 | 107.1 |
| 1 | b3 | 0 | 2 | 1K | 1026 | 8070.0 | 127.1 |
| 1 | b3 | 0 | 3 | 4K | 4098 | 40548.5 | 101.1 |
| 1 | b3 | 1 | 0 | 1K | 1024 | 15453.0 | 66.3 |
| 1 | b3 | 1 | 1 | 4K | 4098 | 30933.3 | 132.5 |
| 1 | b3 | 1 | 2 | 1K | 1026 | 28294.3 | 36.3 |
| 1 | b3 | 1 | 3 | 4K | 4098 | 32493.3 | 126.1 |
| 2 | b3 | 0 | 0 | 1K | 1024 | 7611.1 | 134.5 |
| 2 | b3 | 0 | 1 | 4K | 4098 | 38372.3 | 106.8 |
| 2 | b3 | 0 | 2 | 1K | 1026 | 8069.9 | 127.1 |
| 2 | b3 | 0 | 3 | 4K | 4098 | 40508.5 | 101.2 |
| 2 | b3 | 1 | 0 | 1K | 1024 | 15475.2 | 66.2 |
| 2 | b3 | 1 | 1 | 4K | 4098 | 30923.5 | 132.5 |
| 2 | b3 | 1 | 2 | 1K | 1026 | 28267.2 | 36.3 |
| 2 | b3 | 1 | 3 | 4K | 4098 | 32573.3 | 125.8 |
| 2 | gemm | 0 | 0 | 1K | 1024 | 5685.8 | 180.1 |
| 2 | gemm | 0 | 1 | 4K | 4098 | 32458.0 | 126.3 |
| 2 | gemm | 0 | 2 | 1K | 1026 | 5965.0 | 172.0 |
| 2 | gemm | 0 | 3 | 4K | 4098 | 34145.5 | 120.0 |
| 2 | gemm | 1 | 0 | 1K | 1024 | 15568.4 | 65.8 |
| 2 | gemm | 1 | 1 | 4K | 4098 | 22930.2 | 178.7 |
| 2 | gemm | 1 | 2 | 1K | 1026 | 23862.2 | 43.0 |
| 2 | gemm | 1 | 3 | 4K | 4098 | 23992.4 | 170.8 |

## mean prompt_ms by arm / size / slot (n)

| kind | slot | b3 | gemm | gemm vs b3 |
|---|---|---|---|---|
| 1K | 0 | 7831.8 (6) | 5801.1 (6) | -25.93% |
| 1K | 1 | 21875.0 (6) | 19680.6 (6) | -10.03% |
| 1K | both | 14853.4 (12) | 12740.8 (12) | -14.22% |
| 4K | 0 | 39432.0 (6) | 33240.6 (6) | -15.70% |
| 4K | 1 | 31738.5 (6) | 23443.4 (6) | -26.14% |
| 4K | both | 35585.2 (12) | 28342.0 (12) | -20.35% |

## makespan (total s) per block, paired

| block | b3 | gemm | gemm-b3 s | % |
|---|---|---|---|---|
| 0 | 210.79 | 177.69 | -33.09 | -15.70% |
| 1 | 210.64 | 177.44 | -33.19 | -15.76% |
| 2 | 210.75 | 177.86 | -32.88 | -15.60% |

Paired mean -15.69% (CI95 -15.88..-15.49, n=3)
- b3: makespan mean 210.72 s; prefill_s(server) mean 201.75; peak VRAM 23110.0 MiB
- gemm: makespan mean 177.67 s; prefill_s(server) mean 164.33; peak VRAM 23459.0 MiB

## Engagement (nvprof, gemm arm, prof_gemm session)

- sgemm: 11179 launches, 29159.2 ms (29.2% of 99800 ms GPU)
- masked_softmax: 2560 launches, 7263.2 ms (7.3% of 99800 ms GPU)
- flash_attn_tile: 100 launches, 11815.4 ms (11.8% of 99800 ms GPU)
- flash_attn (any): 230 launches, 11824.0 ms (11.8% of 99800 ms GPU)

Top 12 kernels by time:

- sgemm_128x128x8_NT_vec: n=3990, 16717.0 ms
- [CUDA memcpy HtoD]: n=933, 14801.3 ms
- void flash_attn_tile<int=256, int=256, int=4, int=8, bool=0, int=0>: n=80, 11814.7 ms
- maxwell_sgemm_128x64_nn: n=2560, 11279.8 ms
- ofa_decode_hs_ar_kernel: n=10840, 10072.4 ms
- maxwell_gemm_masked_softmax: n=2560, 7263.2 ms
- void occamy::moe_v2<int=0, int=61>: n=61719, 6828.4 ms
- void mul_mat_q<ggml_type=2, int=64, bool=0, ggml_prec=30, bool=1, bool=1>: n=1170, 4548.2 ms
- occamy_gdn::gdn_fused2: n=13650, 2675.7 ms
- void occamy_gdn::gdn_fused<int=0, int=1, int=1, int=0, int=1>: n=18870, 2274.0 ms
- void mul_mat_q<ggml_type=2, int=16, bool=0, ggml_prec=30, bool=1, bool=1>: n=1170, 1752.4 ms
- void mul_mat_q<ggml_type=2, int=32, bool=0, ggml_prec=30, bool=1, bool=1>: n=1170, 1657.9 ms

server.log (prof_gemm) gemm/fallback lines (1 shown):
    0.54.079.648 I maxwell_gemm: version=4 softmax=online kv_staging_heads=1 call=1 q=1020 k=85248 q_heads=16 kv_heads=2 d=256 mask_stride=170496 key_capacity=98304 splits=21 tail=3328 scratch_bytes=432013888 pool_bytes=4320

server.log (b0_p1_gemm) gemm/fallback lines (1 shown):
    0.45.512.663 I maxwell_gemm: version=4 softmax=online kv_staging_heads=1 call=1 q=1020 k=85248 q_heads=16 kv_heads=2 d=256 mask_stride=170496 key_capacity=98304 splits=21 tail=3328 scratch_bytes=432013888 pool_bytes=4320

Verdict: see engagement counts above; raw N3 analyzer table in N3_TABLE.md.
