#!/usr/bin/env bash
# A5 incremental build: copy of L0 build tree (d4c13eb base), A5 worktree mounted at the card-9 source path inside the container only. CPU only, sm_52.
set -uo pipefail
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114
SRC9=$N2/9/src; SRC=$M/occamy35-48h-20261115/A5/src; OUT=$M/occamy35-48h-20261115/A5/build; LOG=$M/occamy35-48h-20261115/A5/build.log
GITDIR=$N2/7/src/.git
IMG=${IMG:-local/m6000-build-tools:20261013}
CPUS=${CPUS:-0-5}
taskset -c $CPUS nice -n 19 docker run --rm --cpuset-cpus $CPUS --name na5-build -u "$(id -u):$(id -g)" \
  -v "$SRC":"$SRC9":ro -v "$GITDIR":"$GITDIR":ro -v "$OUT":/b -w /b -e TMPDIR=/tmp -e CC=gcc-14 -e CXX=g++-14 -e CUDAHOSTCXX=g++-14 \
  "$IMG" bash -c '
set -e
nice -n 19 cmake --build /b --config Release -j${JOBS:-5} --target llama-server test-backend-ops ggml-cuda
ls -la /b/bin/llama-server /b/bin/*.so | head -30
' > "$LOG" 2>&1
echo "rc=$? log=$LOG"
