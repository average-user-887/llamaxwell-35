# Experimental Maxwell build

This is an investigation build, not a production release or a claim of general quality equivalence. The new GEMM path and mixed-batch cap are both disabled by default. Model weights and user conversation caches are not distributed.

The [tested runtime archive](archives/runtime-01.tar.gz) is 28,635,730 bytes with SHA256 `fa5b6d26482f1837b4a88b888465055413f4d846a81ec2c56d988a0fc82a9260`. Its downloaded GitHub copy and contained CUDA library were hash-verified. See the [publication receipt](PUBLICATION_RECEIPT.json) and [independent package review](PACKAGE_REVIEW.txt). Read [results and limitations](PUBLIC_RESULTS.md) before use.

## What was actually tested

The runtime package combines the existing scheduler candidate server and its matching shared libraries with the exact v4 CUDA library used in the GPU experiments. This is the same overlay used by the deep mixed-serving tests, not a newly compiled monolithic build. The published source combines the two reviewed patches on BUNDLE3, while preserving the previous public audit history.

- Source baseline: `2436f7554c0c8d70afc9acbcce85b2457193a516`.
- Packaged executable platform: Linux x86-64; the test host used NVIDIA driver 580.178.04. This is an observed configuration, not a certified minimum-driver or portable-system requirement.
- CUDA library SHA256: `0c7ffbde334686f17f5ed6a1676c0cd966ec931eaa388900befb06af493330d5`.
- GEMM switch: `GGML_CUDA_FA_MAXWELL_GEMM=1`; omitted or `0` keeps the old dispatcher.
- Scheduler switch: `--mixed-batch-size 1024` or `512`; `0` keeps the old scheduler.
- The scheduler changes `common_params`; keep its matching server/common libraries together. Do not replace only one shared library.
- The rejected head9/speculation prototype is evidence only and is not included in this final source/runtime combination.

The package manifest records each original and published file hash, any redaction, and archive hashes. The CUDA library bytes are unmodified. The package does not include NVIDIA runtime libraries; the tested production image supplied those dependencies. CPU backend plugins are normal llama.cpp dependencies, not authorization for CPU-only model inference; every reported model run offloaded all 41 layers to the M6000.

## Clean source build recipe

The original components were built with CUDA toolkit 12.8.93, GCC/G++ 14 and CMake 3.28.3 in the existing local build image. The following reproduces the relevant configuration from a clean checkout; it is not a claim that a fresh build is byte-reproducible across toolchains, paths or timestamps.

```sh
CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14 cmake -S . -B build-sm52 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=52-real \
  -DCMAKE_CUDA_FLAGS="-Xptxas -v" \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
  -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-sm52 --config Release -j2 --target llama-server test-arg-parser
```

The archived incremental build scripts document what this campaign actually ran; they depend on pre-existing build caches and are not substitutes for the clean recipe. Building does not require GPU access. Model launching must use the operator's approved routing, ownership and memory-admission procedure.

## Evidence limits and rollback

Full probability vectors and repeatability controls show reproducible distribution changes, including KL approximately 0.294 and total variation approximately 20.85% at one 65K checkpoint. Identical top1 choices and successful sample tasks do not establish general model-quality equivalence. See the result report before considering deployment.

The focused scheduler argument tests pass. The broader existing argument test reaches an unrelated HTTP download test and fails in the intentionally network-disabled build environment; its failure log is retained, not relabeled as a full-suite pass.

No production files were replaced during this investigation. Experimental rollback is to omit the new switches or set both to zero, or restore the operator's original matching runtime bundle. CUDA graph experiments use a separate existing presence-based environment flag: setting `GGML_CUDA_MAXWELL_GRAPHS=0` still enables that override; omit it to disable the override.

**Known incompatible option:** leave `GGML_CUDA_MAXWELL_GRAPHS` absent for this runtime. The current paired GDN split path performs a compute-stream synchronization during capture and aborts in the deep two-slot test. Historical graph tests of an earlier build do not qualify this combination. Caching the mutable state-row indices once is not a safe workaround; the graph result report explains the replay hazard.

The archival harnesses retain their original, expired campaign deadline. They are evidence, not unattended launch instructions. A new campaign must establish fresh authorization, resource ownership, route configuration and a bounded deadline without weakening the original records.
