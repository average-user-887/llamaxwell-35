# Card 4 WP2 fused split softmax — RESULT (partial: FP64 RMSE pending)

| arm | N3 makespan s | prefill s | decode tok/s | TTFT mean s | max gap s | single +4K@90K s | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| base gemm (K0 lib) | 178.74 | 164.76 | 16.94 | 22.31 | 11.81 | 23.22 | 23459 |
| fused (WP2) | 165.97 | 141.31 | 18.54 | 19.68 | 10.91 | 21.21 | 23459 |
| delta | -7.1% | -14.2% | +9.4% | -11.8% | -7.6% | -8.6% | 0 |

- Prod-now 177.5 s; single prod 23.09 s. Single 1 block each (n=1, no CI); pair repeat pending in card 7.
- Fused active: log softmax=fused_split_lse, calls=180 softmax_launches=5760.
- Flag: GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1 (with GEMM=1). Branch wp2 88bf35b. Unit tests 18/18.
- Gate: FP64 RMSE vs stock not yet measured. Not adopted yet; goes into card 7 stack.
- Run: <LAB>/occamy35-night-20261114/4/run-20261113T184618Z
