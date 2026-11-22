#!/usr/bin/env bash
# usage: build.sh <variant dir>  -> <dir>/fattn.cu.o, <dir>/out/libggml-cuda.so (+ sha256)
set -euo pipefail
v=$(cd "$1" && pwd -P); B=/tmp/qa-speed-cards/c3/build
proto=~/.gemini/antigravity/brain/5dde1567-112b-432c-a28b-3e2e14940eab/scratch/proto-dev
r2=/media/avg-usr/Storage/m6000-llama-cpp/mtp-moonshot-20261117/sparse/r2
sm=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261114/9/src
img=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
mkdir -p "$v/out"
docker run --rm --pull never --runtime runc --network none --cpuset-cpus 4-5 -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp --tmpfs /tmp:rw,size=512m \
  --user "$(id -u):$(id -g)" -v "$r2/src:$sm:ro" -v "$r2/build:/b:ro" -v "$v:/work" -v "$B:/c3b:ro" -w /b/ggml/src/ggml-cuda \
  --entrypoint /bin/bash "$img" -euc 'bash /c3b/nvcc-fattn.cmd 2>&1 | grep -E "error|warning|spill|maxwell_gemm" | grep -vE "^ptxas info\s+: (Used|Compiling)" | head -60' > "$v/compile.log" 2>&1 || true
test -s "$v/fattn.cu.o"
docker run --rm --pull never --runtime runc --network none --cpuset-cpus 4-5 -e NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" \
  -v "$r2/build:/b:ro" -v "$proto/slim-cuda:/slim:ro" -v "$proto/r3-truncation-a2:/r3fix:ro" -v "$proto/r3-card1:/card1:ro" -v "$proto/r3-pp-kernels:/r3pp:ro" \
  -v "$v:/c3:ro" -v "$v/out:/out" -v "$B:/c3b:ro" -w /b/ggml/src/ggml-cuda --entrypoint /bin/bash "$img" /c3b/link-c3.sh > "$v/link.log" 2>&1
docker run --rm --pull never --runtime runc --network none -e NVIDIA_VISIBLE_DEVICES=void -v "$r2/build:/b:ro" -v "$v/out:/out:ro" --entrypoint /bin/bash "$img" -euc '
 mkdir -p /tmp/s; ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /tmp/s/libcuda.so.1
 LD_LIBRARY_PATH=/out:/tmp/s:/b/bin ldd -r /out/libggml-cuda.so 2>&1 | grep -c "undefined symbol" || true
 /usr/local/cuda/bin/cuobjdump --list-elf /out/libggml-cuda.so | grep -v sm_52 | wc -l' > "$v/verify.log" 2>&1
sha256sum "$v/fattn.cu.o" "$v/out/libggml-cuda.so" | tee "$v/sha256.txt"
