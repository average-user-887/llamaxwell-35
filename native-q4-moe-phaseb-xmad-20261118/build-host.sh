#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
base=../night-native-depth2-q3-80k-r1/libs
taskset -c 2-3 nice -n 19 g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I../sparse/r2/src/ggml/include operator.cpp -L"$base" -lggml -lggml-base -ldl -o operator
LD_LIBRARY_PATH="$base" ./operator --fixture > fixture.txt 2>&1
cat fixture.txt
