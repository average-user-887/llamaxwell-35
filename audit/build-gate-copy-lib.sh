#!/usr/bin/env bash
set -euo pipefail
ROOT=/media/avg-usr/Storage/m6000-llama-cpp
nice -n19 docker run --cpuset-cpus 6-11 --name build-gate-copy-lib-20261013 --network none -v "$ROOT:/w" -w /w/src-b11552-gate-copy-20261013 -e TMPDIR=/w/tmp-combo local/m6000-build-tools:20261013 bash -c '
set -e
export CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release .
nice -n19 cmake --build build --config Release -j6 --target ggml-cuda
' > "$ROOT/native-occamy/gate-copy/build-lib.log" 2>&1
TARGET="$ROOT/prod-libs/libggml-cuda-gate-copy-20261013.so"
test ! -e "$TARGET"
cp "$ROOT/src-b11552-gate-copy-20261013/build/bin/libggml-cuda.so" "$TARGET"
sha256sum "$TARGET" > "$TARGET.sha256"
