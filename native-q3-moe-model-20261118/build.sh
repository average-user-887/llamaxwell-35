#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
src=../sparse/r2/src/ggml
cmp match.inc <(sed -n '/^struct ggml_cuda_occamy_moe_match {/,/^};/p' "$src/src/ggml-cuda/occamy-fused.cuh")
taskset -c 2-3 nice -n 19 g++ -O2 -shared -fPIC -std=c++17 -Wall -Wextra -Werror -I "$src/include" audit.cpp -ldl -o audit.so
nm -D audit.so | grep -F '_Z27ggml_cuda_occamy_moe_launchR25ggml_backend_cuda_contextRK26ggml_cuda_occamy_moe_matchPfi'
objdump -R libs/libggml-cuda.so | grep -F 'JUMP_SLOT' | grep -F '_Z27ggml_cuda_occamy_moe_launchR25ggml_backend_cuda_contextRK26ggml_cuda_occamy_moe_matchPfi'
objdump -R libs/libggml-cuda.so | grep -F JUMP_SLOT | grep -F '_Z33ggml_cuda_occamy_moe_match_windowR25ggml_backend_cuda_contextPK11ggml_cgraphiR26ggml_cuda_occamy_moe_match'
