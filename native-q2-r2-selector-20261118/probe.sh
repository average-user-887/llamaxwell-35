#!/usr/bin/env bash
set -Eeuo pipefail
runtime=${OCCAMY_RUNTIME_DIR:?supply the admitted CUDA library directory}
arm=${1:?supply control or candidate}
output=${2:?supply a fresh synthetic output path}
case "$arm" in
  control) enabled=0 ;;
  candidate) enabled=1 ;;
  *) exit 2 ;;
esac
export LD_LIBRARY_PATH="$runtime"
export OCCAMY_Q2_R2_FIXED_INPUT=1 OCCAMY_Q2_R2_OUTPUT_PATH="$output"
export GGML_CUDA_OCCAMY_FA_MROW=3 GGML_CUDA_OCCAMY_FA2_NU=4
export GGML_CUDA_OCCAMY_FA_SPARSE=1 GGML_CUDA_OCCAMY_FA_SPARSE_R2=1
export GGML_CUDA_OCCAMY_FA_SPARSE_R2_CHECK=0 GGML_CUDA_OCCAMY_FA_SPARSE_PROFILE=0 GGML_CUDA_MV1=0
export GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16="$enabled" GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16_CHECK="$enabled"
sha256sum "$runtime/libggml-cuda.so" "$runtime/test-backend-ops"
exec "$runtime/test-backend-ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p 'kv=81920,nb=2,.*mask=mrow_causal,tail=260' -j 1
