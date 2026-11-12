#!/usr/bin/env bash
# FGDIV server launcher (same flags/env/build as G1b launch-arm.sh). Usage: launch_fgdiv.sh <cap:0|1024> <suffix> [extra env -e args...]  -> container g1c-fgdiv-<suffix>
# Called only by run_all.sh while it holds the GPU lock. G1C_GPU from env. MAXWELL_GRAPHS / FA_MAXWELL_GEMM are never set.
set -euo pipefail
CAP=$1; NAME=g1c-fgdiv-$2; shift 2
M=<LAB>; E=$M/occamy35-day-20261112/evidence/G1b
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
: "${G1C_GPU:?set G1C_GPU}"
ENVS=$(grep -o -e '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-bundle3-20261112.sh | tr '\n' ' ')
echo "$ENVS" | grep -q -E 'MAXWELL_GRAPHS|MAXWELL_GEMM' && { echo "forbidden flag in env"; exit 3; }
[ -z "${GGML_CUDA_MAXWELL_GRAPHS:-}${GGML_CUDA_FA_MAXWELL_GEMM:-}" ] || { echo "forbidden flag in caller env"; exit 3; }
docker run -d --name "$NAME" -p "${G1C_PUBLISH:?set G1C_PUBLISH=host_bind:port}":8080 --device nvidia.com/gpu=$G1C_GPU \
  -v <MODELS>:/models:ro -v $E/slots:/slots \
  -v $E/build/bin:/candidate:ro -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/candidate/libggml-cuda.so:ro \
  -e LD_LIBRARY_PATH=/candidate $ENVS "$@" --entrypoint /candidate/llama-server $IMG \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 \
  --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots -lv 4 \
  --mixed-batch-size "$CAP"
