#!/usr/bin/env bash
# usage: build_alu35.sh <tag> ; builds 35B alu35 worktree ggml-cuda + test-backend-ops -> tmp/libs/libggml-cuda-downx-<tag>.so
set -uo pipefail
TAG=$1
J=/home/avg-usr/.claude/jobs/fc533454/tmp
WT=$J/repos/llamaxwell-35/.claude/worktrees/downx
mkdir -p $J/build-moebw $J/libs $J/logs
LOG=$J/logs/builddownx-$TAG-$(date +%Y%m%dT%H%M%S).log
nice -n 19 docker run --rm --cpuset-cpus 6-11 --name build-downx-$TAG -u $(id -u):$(id -g) \
  -v $WT:/src -v $J/build-moebw:/b -w /src -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 \
  local/m6000-build-tools:20261013 bash -c '
set -e
[ -f /b/CMakeCache.txt ] || cmake -S /src -B /b -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
nice -n 19 cmake --build /b --config Release -j6 --target ggml-cuda test-backend-ops 2>&1 | grep -E "error|warning: .*mmvq|Built target" | tail -20
' > $LOG 2>&1; rc=$?
cp $J/build-moebw/bin/libggml-cuda.so $J/libs/libggml-cuda-downx-$TAG.so && sha256sum $J/libs/libggml-cuda-downx-$TAG.so
echo "rc=$rc log=$LOG"; tail -5 $LOG
