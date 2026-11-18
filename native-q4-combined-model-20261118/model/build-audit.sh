#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
src=../night-cache-only/src
taskset -c 2-3 nice -n 19 g++ -O2 -shared -fPIC -std=c++17 -Wall -Wextra -Werror -I "$src/include" -I "$src/ggml/include" -I "$src/src" audit.cpp -ldl -o audit.so
objdump -R ../night-cache-only/libs/libllama-server-impl.so | grep -F 'JUMP_SLOT  llama_process@'
