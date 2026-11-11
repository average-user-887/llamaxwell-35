# R1 decode profile, prod lib c9825b1 (libggml-cuda-call35-20261111.so), all features ON (DOWNX, QKREUSE, PCSHARE, DOWNH=2)
Method: nvprof --print-gpu-trace in cuda:12.8.1-devel container, `llama bench -fa on -b 4096 -ub 4096 -p 0 -n 6 -d {0,16384} -r 1`, single slot;
one token = kernels between the last two output-head launches (an2.py). Profiled tg: 80.6 t/s d0, 72.0 t/s d16k (profiler-perturbed).
ms/token = sum of kernel time (busy), not wall.

busy ms/token: d0 11.463  d16k 12.935

| kernel | n d0 | ms/tok d0 | share d0 | n 16k | ms/tok 16k | share 16k |
|---|---|---|---|---|---|---|
| occamy::moe_v2<int=0, int=13> | 40 | 4.773 | 41.6% | 40 | 4.778 | 36.9% |
| occamy_gdn::gdn_fused<int=0, int=1, int=1> | 30 | 3.732 | 32.6% | 30 | 3.742 | 28.9% |
| head_rowpack_sm52 | 1 | 1.191 | 10.4% | 1 | 1.185 | 9.2% |
| ofa_decode_kernel | 10 | 0.366 | 3.2% | 10 | 1.772 | 13.7% |
| mul_mat_vec_q<ggml_type=2, int=1, bool=0, bool=0, bool=0, bool=0, bool=1> | 40 | 0.893 | 7.8% | 40 | 0.913 | 7.1% |
| rms_norm_f32<int=256, bool=1, bool=0, bool=0> | 20 | 0.098 | 0.9% | 20 | 0.098 | 0.8% |
| ofa_combine_kernel | 10 | 0.058 | 0.5% | 10 | 0.089 | 0.7% |
| rope_multi<bool=1, bool=0, float> | 20 | 0.074 | 0.6% | 20 | 0.073 | 0.6% |
| rms_norm_mul_q8_1_row<int=1024> | 11 | 0.073 | 0.6% | 11 | 0.073 | 0.6% |
| k_set_rows<float, long, __half> | 20 | 0.064 | 0.6% | 20 | 0.067 | 0.5% |
| _Z11k_bin_bcastIXadL_ZN42_INTERNAL_dee15d3c_11_binbcast_cu_6840010b6op_addEffEEf | 10 | 0.044 | 0.4% | 10 | 0.044 | 0.3% |
| quantize_q8_1 | 10 | 0.035 | 0.3% | 10 | 0.035 | 0.3% |
| unary_gated_op_kernel<__operator_&__ | 10 | 0.032 | 0.3% | 10 | 0.033 | 0.3% |
| gate_copy_sm52 | 10 | 0.021 | 0.2% | 10 | 0.023 | 0.2% |
| k_get_rows_float<float, float> | 2 | 0.009 | 0.1% | 2 | 0.009 | 0.1% |

Attention (ofa_decode + ofa_combine): d0 0.424 ms (3.7%), d16k 1.861 ms (14.4%) vs plan's earlier 1.864 ms at 16K: unchanged.
Depth growth 11.46 -> 12.94 ms/token is almost entirely attention (+1.44 ms); MoE (moe_v2) + GDN + head are 75-80% at either depth.
