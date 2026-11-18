#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
cd "$(dirname "$0")"
here=$(pwd -P)
tmp=$(mktemp -d "$here/.fixture.XXXXXX")
trap 'rmdir -- "$tmp" 2>/dev/null || python3 -c "import shutil,sys;shutil.rmtree(sys.argv[1])" "$tmp"' EXIT
src=${LLAMA_SOURCE:?set pinned LLAMA_SOURCE}
cc=(taskset -c 2-3 nice -n 19 g++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror -I "$src/ggml/include" -I "$src/include" -I "$src/src")
"${cc[@]}" -shared fixture-provider.cpp -o "$tmp/libfakeprocess.so"
"${cc[@]}" -shared fixture-server.cpp -L"$tmp" -lfakeprocess -o "$tmp/libllama-server-impl.so"
"${cc[@]}" fixture-main.cpp -L"$tmp" -Wl,-rpath-link,"$tmp" -lllama-server-impl -o "$tmp/llama-server"
truncate -s 4096 "$tmp/counter"
truncate -s 16384 "$tmp/routes"
timeout -k 1s 15s docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2-3 --read-only --cap-drop ALL -e NVIDIA_VISIBLE_DEVICES=void \
  --user "$(id -u):$(id -g)" -e LD_PRELOAD=/audit.so -e LD_LIBRARY_PATH=/candidate -e NATIVE_Q4_PROMPT_N=80000 \
  -v "$here/audit.so:/audit.so:ro" -v "$tmp:/candidate:ro" -v "$tmp/counter:/native-moe-counters" -v "$tmp/routes:/native-route-counters" \
  --entrypoint /candidate/llama-server sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
python3 -B - "$tmp/routes" <<'PY'
import sys
from pathlib import Path
from struct import unpack

words = unpack("<2048Q", Path(sys.argv[1]).read_bytes())
assert words[:11] == (0x5152344f564c31,1,1,1,512,32,32,1280,0,32,32)
for layer in range(40):
    offset = 16 + 29*layer
    assert words[offset:offset+25] == (0,)*12 + (32,) + (0,)*12
    assert words[offset+25:offset+29] == (256,384,0,0)
print("Device-free preload verified all three hooks, warm sampling, callback detach, padded ID strides and anonymous aggregation.")
PY
