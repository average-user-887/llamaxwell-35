#!/usr/bin/env bash
# CPU-only: compile q8n_rows against the Q8N build libs in the build-tools image (no --gpus). Container nq8n-rowsbuild, --rm.
set -euo pipefail
Q=<LAB>/occamy35-night-20261113/Q8N
docker run --rm --name nq8n-rowsbuild -u "$(id -u):$(id -g)" -v $Q:/q -w /q/gates -e TMPDIR=/tmp local/m6000-build-tools:20261013 \
  g++-14 -O2 -std=c++17 -I /q/src/include -I /q/src/ggml/include q8n_rows.cpp -o q8n_rows -L /q/build/bin -lllama -lggml -lggml-base -Wl,-rpath,/app
sha256sum $Q/gates/q8n_rows
