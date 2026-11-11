# Stage B: Exactness First Verification Report

Generated at: 2026-11-11T00:27:22.818351+00:00

Deterministic greedy generation (>=512 tokens, temp 0, top_k 1, n_probs 5, max logprob diff == 0, zero NaNs).

## Reference Arm Check (Flag-OFF vs build-clean)

| Lib Key | Library Path | 512 Match | Max Diff | NaNs | Clean Confirmed |
|---|---|---:|---:|---:|:---:|
| c1011h | `libggml-cuda-c1011h.so` | Y | 0.000000 | 0 | **PASS** |
| c11h | `libggml-cuda-c11h.so` | Y | 0.000000 | 0 | **PASS** |
| call | `libggml-cuda-call.so` | Y | 0.000000 | 0 | **PASS** |

## Candidate Arms Exactness

| Candidate Arm | Description | 512 Match | Max Diff | NaNs | Long Rep 1 (4096) | Long Rep 2 (4096) | 2-Slot Conc | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| c11h | GDN Phase-C PCSHARE=1 (lib c11h, Patch H1) | Y | 0.000000 | 0 | Y | Y | Y | **PASS** |
| c1011h_b | GDN Phase-B QKREUSE=1 (lib c1011h) | Y | 0.000000 | 0 | N/A | N/A | N/A | **PASS** |
| c1011h_c | GDN Phase-C PCSHARE=1 (lib c1011h) | Y | 0.000000 | 0 | Y | Y | Y | **PASS** |
| c1011h_bc | GDN Phase-B QKREUSE=1 + Phase-C PCSHARE=1 (lib c1011h) | Y | 0.000000 | 0 | Y | Y | Y | **PASS** |
| call_dh2 | Combo: GDN (QKREUSE+PCSHARE) + MoE DOWNH=2 (lib call) | Y | 0.000000 | 0 | Y | Y | Y | **PASS** |
| call_dh1 | Combo: GDN (QKREUSE+PCSHARE) + MoE DOWNH=1 (lib call) | Y | 0.000000 | 0 | Y | Y | Y | **PASS** |

