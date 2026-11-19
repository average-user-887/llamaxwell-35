#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "$0")"
here=$(pwd -P)
c=$(dirname -- "$(dirname -- "$here")")
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
timeout -k 2s 120s taskset -c 2-3 nice -n 19 docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2-3 --tmpfs /tmp:rw,size=512m --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp \
 -v "$c/sparse/r2/build:/b:ro" -v "$c/finish33-r2-stride/fattn-occamy.cu.o:/stride/fattn-occamy.cu.o:ro" \
 -v "$c/resume-q2-moe-cohorts/occamy-fused.cu.o:/control/occamy-fused.cu.o:ro" -v "$c/night-gdn-t2-q8-fast/occamy-gdn.cu.o:/q8/occamy-gdn.cu.o:ro" \
 -v "$here:/work" -w /work --entrypoint /bin/bash "$image" -euc 'nice -n 19 bash /work/compile.sh' > relink.log 2>&1
cmp control-relinked.so "$c/finish33-r2-stride/libggml-cuda.so"
