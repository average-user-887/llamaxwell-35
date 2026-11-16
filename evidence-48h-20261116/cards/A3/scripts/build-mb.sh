#!/usr/bin/env bash
# Build the A3 microbench CPU-side (no GPU): extract the occamy GDN kernels from the A3 worktree and compile for sm_52.
set -euo pipefail
A=<W>/A3
F=$A/src/ggml/src/ggml-cuda/occamy-gdn.cu
awk '/^namespace occamy_gdn \{/{on=1} /^struct DevState/{on=0} on' "$F" > $A/tools/a3_kern.inc
echo '} // namespace occamy_gdn' >> $A/tools/a3_kern.inc
docker run --rm --name a3-mb-build -u "$(id -u):$(id -g)" --cpuset-cpus 0-5 -e TMPDIR=/tmp -v $A/tools:/t -w /t --entrypoint bash \
  local/m6000-build-tools:20261013 -c 'nvcc -O3 -std=c++17 -use_fast_math "--generate-code=arch=compute_52,code=[sm_52]" -Xptxas -v -o a3_mb a3_mb.cu 2>&1 | grep -A2 "gdn_fused2" | grep -v "^--"; ls -la a3_mb'
