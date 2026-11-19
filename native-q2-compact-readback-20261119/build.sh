#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
card=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
upstream=$(dirname -- "$card")/night-one-graph-draft
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
cd -- "$card"
sha256sum --check --quiet INPUTS.sha256
PYTHONDONTWRITEBYTECODE=1 python3 prepare.py > mask-check.txt
timeout -k 3s 80s docker run --rm --pull never --runtime runc --network none --cpuset-cpus 2 \
  --name "nmmoon-compact-build-$$" --tmpfs /tmp:rw,size=256m --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e TMPDIR=/tmp -v "$card:/work" -v "$upstream/src:/src:ro" \
  -v "$upstream/build/bin:/runtime:ro" -w /work --entrypoint /bin/bash "$image" -euc '
    /usr/local/cuda/bin/nvcc --version
    /usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG \
      --generate-code=arch=compute_52,code=sm_52 -Xptxas=-v -Xcompiler=-rdynamic \
      -I/src/common -I/src/include -I/src/ggml/include \
      -L/runtime -l:libllama-common.so.0.6.0 -l:libllama.so.0.6.0 -l:libggml.so.0.26.0 \
      -l:libggml-base.so.0.26.0 -o compact-readback readback.cu
    /usr/local/cuda/bin/cuobjdump --list-elf compact-readback
    LD_LIBRARY_PATH=/runtime ./compact-readback --cpu-check
  ' > build.log 2>&1
