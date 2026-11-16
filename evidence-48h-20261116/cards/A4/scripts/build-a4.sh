#!/usr/bin/env bash
# A4 MoE FP32 prefill (from R0 tree) incremental build: copy of the batch8 (card 9) build tree, with the 28h worktree mounted at the card-9
# source path inside the container only, so make rebuilds only the touched ggml-cuda units and relinks. CPU only, sm_52, no --gpus.
set -uo pipefail
N2=<M6000_ROOT>/occamy35-night-20261114
SRC9=$N2/9/src; SRC=<W>/A4/src; OUT=<W>/A4/build; LOG=<W>/A4/build.log
GITDIR=$N2/7/src/.git
IMG=${IMG:-local/m6000-build-tools:20261013}
CPUS=${CPUS:-0-5}
taskset -c $CPUS nice -n 19 docker run --rm --cpuset-cpus $CPUS --name na4-build -u "$(id -u):$(id -g)" \
  -v "$SRC":"$SRC9":ro -v "$GITDIR":"$GITDIR":ro -v "$OUT":/b -w /b -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 \
  "$IMG" bash -c '
set -e
nvcc --version | tail -2
nice -n 19 cmake --build /b --config Release -j${JOBS:-5} --target llama-server test-backend-ops ggml-cuda
ls -la /b/bin/llama-server /b/bin/test-backend-ops /b/bin/*.so | head -30
' > "$LOG" 2>&1
echo "rc=$? log=$LOG"
