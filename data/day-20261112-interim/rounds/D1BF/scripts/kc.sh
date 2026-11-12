#!/usr/bin/env bash
# quick ptxas-only compile of a scratch copy of occamy-gdn.cu: kc.sh <file.cu> [extra nvcc -D...]
SRC=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/cand/D1BF
F=$(readlink -f "$1"); shift
taskset -c 6-11 docker run --rm --cpuset-cpus 6-11 --name kc-d1bf -u "$(id -u):$(id -g)" -v "$SRC":/src:ro -v "$(dirname $F)":/w -w /w -e TMPDIR=/tmp \
  local/m6000-build-tools:20261013 nvcc -ccbin g++-14 -I/src/ggml/src/ggml-cuda -I/src/ggml/src -I/src/ggml/include -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_BUILD -DGGML_SHARED \
  -Xptxas -v -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_52,code=[sm_52] -Xcompiler=-fPIC -use_fast_math -extended-lambda "$@" -c "$(basename $F)" -o /tmp/x.o 2>&1 | grep -A2 gdn_fused2 | grep -E "registers|spill|error"
