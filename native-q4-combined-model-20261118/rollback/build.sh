#!/usr/bin/env bash
set -euo pipefail
umask 077
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base=$(dirname -- "$here")/night-cache-only
image=sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed
timeout --signal=TERM --kill-after=5s 150s taskset -c 2-3 nice -n 19 docker run --rm --pull never \
  --runtime runc --network none --cpuset-cpus 2-3 --memory 600m --memory-swap 600m \
  --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e LD_LIBRARY_PATH=/candidate \
  -v "$base/libs:/candidate:ro" -v "$base/src/include:/llama:ro" \
  -v "$base/src/ggml/include:/ggml:ro" -v "$here:/work" -w /work \
  --entrypoint /bin/bash "$image" -euc '
    nice -n 19 g++-14 -std=c++17 -O2 -Wall -Wextra -Werror -I/llama -I/ggml state.cpp \
      -L/candidate -Wl,-rpath-link,/candidate -Wl,--export-dynamic-symbol=ggml_backend_sched_graph_compute_async -Wl,--export-dynamic-symbol=_Z23ggml_cuda_occamy_gdn_t2* -Wl,--export-dynamic-symbol=_Z28ggml_cuda_op_gated_delta_net* -Wl,--export-dynamic-symbol=_Z40ggml_cuda_op_gated_delta_net_fused_cache* -lllama -lggml -lggml-base -ldl -o state.build
    nice -n 19 ./state.build --self-test
    mv -f state.build state
  '
