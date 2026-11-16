#!/usr/bin/env bash
# Build the A2 microbench CPU-side (no GPU): extract the occamy kernels from the A2 worktree and compile for sm_52.
set -euo pipefail
A=<W>/A2
F=$A/src/ggml/src/ggml-cuda/occamy-fused.cu
awk '/^namespace occamy \{/{on=1} /^struct DevState/{on=0} on' "$F" > $A/tools/moe3_kern.inc
echo '} // namespace occamy' >> $A/tools/moe3_kern.inc
docker run --rm --name a2-mb-build -u "$(id -u):$(id -g)" --cpuset-cpus 0-5 -e TMPDIR=/tmp -v $A/tools:/t -w /t --entrypoint bash \
  local/m6000-build-tools:20261013 -c 'nvcc -O3 -std=c++17 -use_fast_math -extended-lambda "--generate-code=arch=compute_52,code=[sm_52]" -Xptxas -v -o moe3_mb moe3_mb.cu 2>&1 | grep -A1 "moe_v3" | grep -v "^--"; ls -la moe3_mb'
