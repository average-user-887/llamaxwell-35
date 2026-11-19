#!/usr/bin/env bash
# Portable CPU-only build. Run inside the documented pinned compiler environment.
set -Eeuo pipefail
umask 077
cd -- "$(dirname -- "$0")"
: "${LLAMA_HEADERS:?set to the pinned N40 include directory}"
: "${GGML_HEADERS:?set to the pinned GGML include directory}"
: "${CORE_LIBS:?set to the directory with the exact compatible link libraries}"
g++-14 -std=c++17 -O2 -Wall -Wextra -Werror state.cpp \
  -I"$LLAMA_HEADERS" -I"$GGML_HEADERS" \
  -L"$CORE_LIBS" -Wl,-rpath-link,"$CORE_LIBS" \
  -Wl,--export-dynamic-symbol=ggml_backend_sched_graph_compute_async \
  -Wl,--export-dynamic-symbol=_Z23ggml_cuda_occamy_gdn_t2* \
  -Wl,--export-dynamic-symbol=_Z26ggml_cuda_occamy_fa_sparse* \
  -Wl,--export-dynamic-symbol=_Z28ggml_cuda_op_gated_delta_net* \
  -Wl,--export-dynamic-symbol=_Z40ggml_cuda_op_gated_delta_net_fused_cache* \
  -lllama -lggml -lggml-base -ldl -o state
LD_LIBRARY_PATH="$CORE_LIBS" ./state --self-test
