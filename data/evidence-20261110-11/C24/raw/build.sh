#!/usr/bin/env bash
# C24 incremental build (build-clean tree copied, sources back-dated except occamy-gdn.cu, occamy-fused.cu). CPU only, no GPU.
# Recipe follows established build_b11552_*.sh / downx scripts/build.sh (cmake flags identical) plus -Xptxas -v for resource report.
# Tool image: local/m6000-build-tools:20261013 (nvcc 12.8.93, gcc-14 14.2.0, cmake 3.28.3). Alternative: nvidia/cuda:12.8.1-devel-ubuntu24.04 (nvcc 12.8.93 also; needs apt gcc-14 cmake).
# Source mounted READ-ONLY; output only under WORK/build-clean.
set -uo pipefail
WORK=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110
SRC=${SRC:-$WORK/cand/C24}
OUT=${OUT:-$WORK/cand/C24/build}
IMG=${IMG:-local/m6000-build-tools:20261013}
LOG=$WORK/results/C24/raw/build.log
mkdir -p "$OUT"
taskset -c 6-11 nice -n 19 docker run --rm --cpuset-cpus 6-11 --name build-c24 -u "$(id -u):$(id -g)" \
  -v "$SRC":/src:ro -v "$OUT":/b -w /b -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 \
  "$IMG" bash -c '
set -e
nvcc --version | tail -2; gcc-14 --version | head -1; cmake --version | head -1
[ -f /b/CMakeCache.txt ] || cmake -S /src -B /b -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_FLAGS="-Xptxas -v"
nice -n 19 cmake --build /b --config Release -j3 --target ggml-cuda
ls -la /b/bin/libggml-cuda.so; sha256sum /b/bin/libggml-cuda.so
' > "$LOG" 2>&1
echo "rc=$? log=$LOG"
