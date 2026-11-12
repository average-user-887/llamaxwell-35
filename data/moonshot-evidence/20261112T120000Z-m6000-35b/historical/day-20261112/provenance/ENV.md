# Environment and provenance (12 Nov 2026)

| Item | Value |
|---|---|
| GPU | Quadro M6000 24 GB, sm_52, 24 SMs, VBIOS 84.00.4C.00.03 |
| Driver | 580.178.04 (nvidia-smi CUDA 13.0) |
| Power / clocks | 250 W (default = max; min 150 W); app clocks 1152 / 3305 MHz (setting not active) |
| Host kernel | 7.0.0-42-generic |
| Build image | local/m6000-build-tools:20261013; CUDA 12.8.1 (nvcc V12.8.93), gcc 14.2.0, cmake 3.28.3 |
| CMake | GGML_CUDA=ON, GGML_NATIVE=OFF, BACKEND_DL=ON, CPU_ALL_VARIANTS=ON, CUDA_ARCHITECTURES=52-real, Release, `-Xptxas -v` |
| Version | llama.cpp 0.6.0-dev, ggml 0.26.0 |
| Fork point | upstream ggml-org/llama.cpp b11552 (23b0202a189c44a54625aadcb37a946dd1d6278d); from repo history root, no merge-base |
| Runtime image | local/llama.cpp:server-cuda-b11552-sm52-normq8 (sha256:3a0caa9a...), prod libggml-cuda.so bind-mounted read-only |
| Model | `<MODELS>/occamy-1.0-Q4_0-headq40b.gguf`, 19583940352 B, Q4_0, sha256 2b5d7a2c...d9cc0 |
| Server | -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 -fa on --cache-ram 12288 --ctx-checkpoints 16 |

| Variant | Commit | Lib sha256 (prefix) | Env gates added |
|---|---|---|---|
| a baseline | c9825b1 | c3d0ecd0 | full baseline gate set (see env.json) |
| b BUNDLE2 | 2ca7934 | c7d0455e | FA_PIPE=3, MOE_LOP3, MOE_TAILBAL, GDN_PREF, HOSTGAP, SMALLFUSE=2 |
| c GDN4 | 85dbeed | 31181041 | GDN_REPACK |
| d BUNDLE3 | 2436f75 | 94c777dc | N2SPLIT, FA_ADDR, N2GDN |

Full hashes, env strings and app binary hashes: `env.json`. Paths: `<LAB>` = lab root, `<MODELS>` = model store.
