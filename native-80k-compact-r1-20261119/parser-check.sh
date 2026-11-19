#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
c=$(dirname -- "$here")
n40=$c/night-one-graph-draft
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
cd "$here"
timeout -k 3s 40s docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2 \
  --name "nmmoon-compact-parser-r1-$$" --tmpfs /tmp:rw,size=128m --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp -e LD_LIBRARY_PATH=/runtime \
  -v "$n40/src:/src:ro" -v "$n40/build/bin:/runtime:ro" -v "$here:/work" -w /work \
  --entrypoint /bin/bash "$image" -euc '
    /usr/bin/g++-14 -std=c++17 -O2 -Wall -Wextra -Werror -I/src/common -I/src/include -I/src/ggml/include \
      parser-smoke.cpp -L/runtime -l:libllama-common.so.0.6.0 -l:libllama.so.0.6.0 -Wl,-rpath-link,/runtime -o parser-smoke
    ./parser-smoke --spec-type draft-mtp --spec-draft-n-max 1
  ' > parser-check.log 2>&1
