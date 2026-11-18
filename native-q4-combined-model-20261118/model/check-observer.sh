#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
here=$(pwd -P)
tmp=$(mktemp -d "$here/cpu-fixture/tmp.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
src=../night-cache-only/src
cc=(taskset -c 2-3 nice -n 19 g++ -std=c++17 -O2 -fPIC -I "$src/ggml/include" -I "$src/include" -I "$src/src")
"${cc[@]}" -shared cpu-fixture/mock.cpp -o "$tmp/libggml-cuda.so"
"${cc[@]}" -shared cpu-fixture/provider.cpp -o "$tmp/libfakeprocess.so" -ldl
"${cc[@]}" -shared cpu-fixture/server.cpp -L"$tmp" -lfakeprocess -o "$tmp/libllama-server-impl.so"
"${cc[@]}" cpu-fixture/main.cpp -L"$tmp" -Wl,-rpath-link,"$tmp" -lllama-server-impl -o "$tmp/llama-server"
truncate -s 4096 "$tmp/counter"
timeout -k 1s 15s docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2-3 --read-only --cap-drop ALL -e NVIDIA_VISIBLE_DEVICES=void \
  --user "$(id -u):$(id -g)" -e LD_PRELOAD=/audit.so -e LD_LIBRARY_PATH=/candidate -e NATIVE_Q4_PROMPT_N=2000 \
  -v "$here/audit.so:/audit.so:ro" -v "$tmp:/candidate:ro" -v "$tmp/counter:/native-moe-counters" \
  --entrypoint /candidate/llama-server sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
python3 -B - "$tmp/counter" <<'PY'
from pathlib import Path
from observation import snapshot
import sys
w=snapshot(Path(sys.argv[1]))
assert w[2:11]==(1,80,80,80,80,0,0,2,0) and w[16:56]==(2,)*40 and w[64:104]==w[112:152]==(1,)*40 and w[160:167]==(0,1,1,0,0,1,1), w
print("CPU fixture confirmed actual preloaded server-call interception, prompt positions, post-prompt positions, non-target fallback and forty layer counters.")
PY
