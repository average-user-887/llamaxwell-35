| arm | append | rep | wall s (HTTP) | prefill s (server) | GPU span s | GPU busy s | kernels |
|---|---|---|---|---|---|---|---|
| A | +1K | 1 | 8.00 | 7.97 | 7.94 | 7.91 | 4186 |
| A | +4K | 1 | 32.01 | 31.98 | 31.96 | 31.90 | 11443 |
| A | +1K | 2 | 8.01 | 7.99 | 7.95 | 7.92 | 4186 |
| A | +4K | 2 | 32.00 | 31.98 | 31.96 | 31.92 | 11443 |
| B | +1K | 1 | 5.87 | 5.84 | 5.81 | 5.77 | 5806 |
| B | +4K | 1 | 23.42 | 23.39 | 23.36 | 23.30 | 17573 |
| B | +1K | 2 | 5.90 | 5.88 | 5.84 | 5.81 | 5806 |
| B | +4K | 2 | 23.47 | 23.44 | 23.42 | 23.37 | 17573 |

### Grouped shares of GPU busy time in prefill (% ; ms)

| arm | append | rep | FA/attention | MoE matmul | dense mmq/mmvq | GDN/ssm | quantize/copy | other |
|---|---|---|---|---|---|---|---|---|
| A | +1K | 1 | 77.7% (6143) | 10.5% (830) | 7.5% (593) | 1.5% (121) | 1.7% (133) | 1.1% (87) |
| A | +1K | 2 | 77.7% (6154) | 10.5% (830) | 7.5% (594) | 1.5% (121) | 1.7% (132) | 1.1% (88) |
| A | +4K | 1 | 78.3% (24968) | 10.3% (3282) | 7.3% (2341) | 1.5% (480) | 1.5% (490) | 1.1% (343) |
| A | +4K | 2 | 78.3% (24994) | 10.3% (3276) | 7.3% (2333) | 1.5% (480) | 1.5% (490) | 1.1% (343) |
| B | +1K | 1 | 0.4% (25) | 14.2% (820) | 79.5% (4586) | 2.1% (121) | 2.3% (133) | 1.5% (88) |
| B | +1K | 2 | 0.4% (25) | 14.3% (829) | 79.4% (4615) | 2.1% (121) | 2.3% (133) | 1.5% (88) |
| B | +4K | 1 | 0.1% (26) | 14.0% (3263) | 80.2% (18697) | 2.1% (481) | 2.1% (492) | 1.5% (344) |
| B | +4K | 2 | 0.1% (26) | 14.0% (3281) | 80.2% (18747) | 2.1% (482) | 2.1% (492) | 1.5% (344) |

### Top 10 kernels, arm A, +1K (rep1 | rep2 ms)

| # | kernel | group | rep1 ms | rep2 ms | % of prefill GPU busy (rep1) | % of prefill wall (rep1) | calls |
|---|---|---|---|---|---|---|---|
| 1 | `flash_attn_tile<int=256, int=256, int=4, int=8, bool=0, int=0>` | FA/attention | 6139.5 | 6150.7 | 77.6 | 76.8 | 20 |
| 2 | `sgemm_128x128x8_NT_vec` | dense mmq/mmvq | 524.2 | 524.6 | 6.6 | 6.6 | 149 |
| 3 | `mul_mat_q<ggml_type=2, int=64, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 456.7 | 456.5 | 5.8 | 5.7 | 117 |
| 4 | `mul_mat_q<ggml_type=2, int=32, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 190.0 | 189.3 | 2.4 | 2.4 | 117 |
| 5 | `mul_mat_q<ggml_type=2, int=16, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 155.1 | 155.4 | 2.0 | 1.9 | 117 |
| 6 | `gated_delta_net_cuda<int=128, bool=0, bool=0, int=4>` | GDN/ssm | 108.3 | 108.2 | 1.4 | 1.4 | 60 |
| 7 | `dequantize_block_q4_0<float>` | quantize/copy | 73.2 | 72.7 | 0.9 | 0.9 | 307 |
| 8 | `maxwell_sgemm_128x64_tn` | dense mmq/mmvq | 42.8 | 42.7 | 0.5 | 0.5 | 98 |
| 9 | `rms_norm_f32<int=256, bool=1, bool=0, bool=0>` | other | 22.6 | 22.7 | 0.3 | 0.3 | 100 |
| 10 | `rms_norm_f32<int=256, bool=0, bool=0, bool=1>` | other | 18.8 | 18.8 | 0.2 | 0.2 | 120 |
(prefill GPU idle gaps rep1: 0.03 s of 7.94 s span)

### Top 10 kernels, arm A, +4K (rep1 | rep2 ms)

| # | kernel | group | rep1 ms | rep2 ms | % of prefill GPU busy (rep1) | % of prefill wall (rep1) | calls |
|---|---|---|---|---|---|---|---|
| 1 | `flash_attn_tile<int=256, int=256, int=4, int=8, bool=0, int=0>` | FA/attention | 24952.9 | 24978.8 | 78.2 | 78.0 | 50 |
| 2 | `sgemm_128x128x8_NT_vec` | dense mmq/mmvq | 2096.4 | 2088.3 | 6.6 | 6.5 | 596 |
| 3 | `mul_mat_q<ggml_type=2, int=64, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 1835.9 | 1831.3 | 5.8 | 5.7 | 468 |
| 4 | `mul_mat_q<ggml_type=2, int=32, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 711.4 | 710.3 | 2.2 | 2.2 | 468 |
| 5 | `mul_mat_q<ggml_type=2, int=16, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 650.8 | 650.3 | 2.0 | 2.0 | 468 |
| 6 | `gated_delta_net_cuda<int=128, bool=0, bool=0, int=4>` | GDN/ssm | 430.8 | 430.6 | 1.4 | 1.3 | 150 |
| 7 | `dequantize_block_q4_0<float>` | quantize/copy | 290.3 | 290.2 | 0.9 | 0.9 | 1228 |
| 8 | `maxwell_sgemm_128x64_tn` | dense mmq/mmvq | 170.6 | 170.5 | 0.5 | 0.5 | 392 |
| 9 | `rms_norm_f32<int=256, bool=1, bool=0, bool=0>` | other | 89.8 | 89.8 | 0.3 | 0.3 | 250 |
| 10 | `rms_norm_f32<int=256, bool=0, bool=0, bool=1>` | other | 74.6 | 74.6 | 0.2 | 0.2 | 300 |
(prefill GPU idle gaps rep1: 0.06 s of 31.96 s span)

### Top 10 kernels, arm B, +1K (rep1 | rep2 ms)

| # | kernel | group | rep1 ms | rep2 ms | % of prefill GPU busy (rep1) | % of prefill wall (rep1) | calls |
|---|---|---|---|---|---|---|---|
| 1 | `sgemm_128x128x8_NT_vec` | dense mmq/mmvq | 2044.4 | 2064.4 | 35.4 | 34.9 | 469 |
| 2 | `maxwell_sgemm_128x64_nn` | dense mmq/mmvq | 1464.8 | 1472.7 | 25.4 | 25.0 | 320 |
| 3 | `maxwell_gemm_masked_softmax` | dense mmq/mmvq | 945.6 | 944.3 | 16.4 | 16.1 | 320 |
| 4 | `mul_mat_q<ggml_type=2, int=64, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 449.1 | 456.4 | 7.8 | 7.7 | 117 |
| 5 | `mul_mat_q<ggml_type=2, int=32, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 186.3 | 186.9 | 3.2 | 3.2 | 117 |
| 6 | `mul_mat_q<ggml_type=2, int=16, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 155.9 | 156.4 | 2.7 | 2.7 | 117 |
| 7 | `gated_delta_net_cuda<int=128, bool=0, bool=0, int=4>` | GDN/ssm | 108.2 | 108.4 | 1.9 | 1.8 | 60 |
| 8 | `dequantize_block_q4_0<float>` | quantize/copy | 72.8 | 73.4 | 1.3 | 1.2 | 307 |
| 9 | `maxwell_sgemm_128x64_tn` | dense mmq/mmvq | 42.3 | 43.2 | 0.7 | 0.7 | 98 |
| 10 | `sgemm_32x32x32_NN_vec` | dense mmq/mmvq | 25.3 | 25.9 | 0.4 | 0.4 | 320 |
(prefill GPU idle gaps rep1: 0.03 s of 5.81 s span)

### Top 10 kernels, arm B, +4K (rep1 | rep2 ms)

| # | kernel | group | rep1 ms | rep2 ms | % of prefill GPU busy (rep1) | % of prefill wall (rep1) | calls |
|---|---|---|---|---|---|---|---|
| 1 | `sgemm_128x128x8_NT_vec` | dense mmq/mmvq | 8319.5 | 8352.0 | 35.7 | 35.5 | 1876 |
| 2 | `maxwell_sgemm_128x64_nn` | dense mmq/mmvq | 5932.4 | 5947.0 | 25.5 | 25.3 | 1280 |
| 3 | `maxwell_gemm_masked_softmax` | dense mmq/mmvq | 3910.7 | 3911.0 | 16.8 | 16.7 | 1280 |
| 4 | `mul_mat_q<ggml_type=2, int=64, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 1817.0 | 1830.5 | 7.8 | 7.8 | 468 |
| 5 | `mul_mat_q<ggml_type=2, int=32, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 705.8 | 710.7 | 3.0 | 3.0 | 468 |
| 6 | `mul_mat_q<ggml_type=2, int=16, bool=0, ggml_prec=30, bool=1, bool=1>` | MoE matmul | 655.4 | 656.2 | 2.8 | 2.8 | 468 |
| 7 | `gated_delta_net_cuda<int=128, bool=0, bool=0, int=4>` | GDN/ssm | 431.7 | 432.5 | 1.9 | 1.8 | 150 |
| 8 | `dequantize_block_q4_0<float>` | quantize/copy | 291.8 | 291.5 | 1.3 | 1.2 | 1228 |
| 9 | `maxwell_sgemm_128x64_tn` | dense mmq/mmvq | 170.0 | 170.1 | 0.7 | 0.7 | 392 |
| 10 | `sgemm_32x32x32_NN_vec` | dense mmq/mmvq | 138.8 | 140.8 | 0.6 | 0.6 | 960 |
(prefill GPU idle gaps rep1: 0.06 s of 23.36 s span)

### Restore and first-decode-step (1-token batch) phases

| arm | cell | restore HTTP s | restore GPU busy s | restore kernels/copies | 1-token step HTTP s | 1-token step GPU busy ms | top kernel of 1-token step |
|---|---|---|---|---|---|---|---|
| A | r1_1K | 0.56 | 0.150 | 80 | 0.100 | 23.0 | `ofa_decode_hs_ar_kernel(ofa_params)` 6.7 ms |
| A | r1_4K | 0.58 | 0.156 | 80 | 0.103 | 23.4 | `ofa_decode_hs_ar_kernel(ofa_params)` 7.0 ms |
| A | r2_1K | 0.56 | 0.150 | 80 | 0.096 | 22.9 | `ofa_decode_hs_ar_kernel(ofa_params)` 6.7 ms |
| A | r2_4K | 0.56 | 0.152 | 80 | 0.101 | 23.3 | `ofa_decode_hs_ar_kernel(ofa_params)` 7.0 ms |
| B | r1_1K | 0.57 | 0.156 | 80 | 0.103 | 23.1 | `ofa_decode_hs_ar_kernel(ofa_params)` 6.8 ms |
| B | r1_4K | 0.62 | 0.168 | 80 | 0.099 | 23.0 | `ofa_decode_hs_ar_kernel(ofa_params)` 7.0 ms |
| B | r2_1K | 0.54 | 0.150 | 80 | 0.096 | 22.8 | `ofa_decode_hs_ar_kernel(ofa_params)` 6.7 ms |
| B | r2_4K | 0.55 | 0.154 | 80 | 0.099 | 23.3 | `ofa_decode_hs_ar_kernel(ofa_params)` 7.0 ms |
