#!/usr/bin/env bash
# Q8N: 2436f75 + native q8_0 KV occamy decode (no cap patch). CPU only, sm_52, container nq8n-build, no --gpus.
# Mirrors ../E5/build-e5.sh; builds llama-server + ggml-cuda + cpu variants. Src mounted read-only.
set -uo pipefail
E=<LAB>/occamy35-night-20261113/Q8N
SRC=$E/src; OUT=$E/build; LOG=$E/build.log
IMG=${IMG:-local/m6000-build-tools:20261013}
mkdir -p "$OUT"
taskset -c 6-11 nice -n 19 docker run --rm --cpuset-cpus 6-11 --name nq8n-build -u "$(id -u):$(id -g)" \
  -v "$SRC":/src:ro -v "$OUT":/b -w /b -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 \
  "$IMG" bash -c '
set -e
nvcc --version | tail -2; gcc-14 --version | head -1; cmake --version | head -1
[ -f /b/CMakeCache.txt ] || cmake -S /src -B /b -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_FLAGS="-Xptxas -v"
nice -n 19 cmake --build /b --config Release -j3 --target llama-server test-arg-parser ggml-cuda $(cmake --build /b --target help | grep -o "ggml-cpu-[a-z0-9_.-]*" | sort -u | tr "\\n" " ")
ls -la /b/bin/llama-server /b/bin/*.so | head -30
' > "$LOG" 2>&1
echo "rc=$? log=$LOG"
