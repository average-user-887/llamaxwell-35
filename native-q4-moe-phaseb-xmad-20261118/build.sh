#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
here=$(pwd -P)
base=$(dirname "$here")/sparse/r2
n98=$(dirname "$here")/resume-q4-r2-sparse
source=@PRIVATE_COMPILER_SOURCE_MOUNT@
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
python3 -B prepare.py
timeout -k 2s 480s taskset -c 2-3 nice -n 19 docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2-3 --tmpfs /tmp:rw,size=512m --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp -v "$base/src:$source:ro" -v "$here/occamy-fused.cu:$source/ggml/src/ggml-cuda/occamy-fused.cu:ro" -v "$base/build:/b:ro" -v "$here/../resume-q4-moe-cohorts/occamy-fused.cu.o:/q3/occamy-fused.cu.o:ro" -v "$here:/work" -v "$n98/fattn-occamy.cu.o:/n98/fattn-occamy.cu.o:ro" -w /work --entrypoint /bin/bash "$image" -euc 'nice -n 19 bash /work/compile.sh' > build.log 2>&1
cmp control-relinked.so "$here/../resume-q4-combined-native/libggml-cuda.so"
