#!/usr/bin/env bash
# G1b server launcher. Usage: launch-arm.sh <cap:0|1024> <name-suffix>   -> container g1b-<suffix>, 127.0.0.1:18493
# Called only by run-g1b.sh while it holds the GPU lock. GPU id comes from env G1B_GPU (not stored in files).
# cache-ram 0 (as Codex deep-mixed): stops host prompt-cache hits across units. Env = BUNDLE3 prod GGML_* from run-35b-bundle3-20261112.sh, verbatim. GGML_CUDA_MAXWELL_GRAPHS and GGML_CUDA_FA_MAXWELL_GEMM are never set.
set -euo pipefail
CAP=$1; NAME=g1b-$2
M=<LAB>; E=$M/occamy35-day-20261112/evidence/G1b
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
: "${G1B_GPU:?set G1B_GPU}"
ENVS=$(grep -o -e '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-bundle3-20261112.sh | tr '\n' ' ')
echo "$ENVS" | grep -q -E 'MAXWELL_GRAPHS|MAXWELL_GEMM' && { echo "forbidden flag in env"; exit 3; }
[ -z "${GGML_CUDA_MAXWELL_GRAPHS:-}${GGML_CUDA_FA_MAXWELL_GEMM:-}" ] || { echo "forbidden flag in caller env"; exit 3; }
docker run -d --name "$NAME" -p 127.0.0.1:18493:8080 --device nvidia.com/gpu=$G1B_GPU \
  -v <MODELS>:/models:ro -v $E/slots:/slots \
  -v $E/build/bin:/candidate:ro -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/candidate/libggml-cuda.so:ro \
  -e LD_LIBRARY_PATH=/candidate $ENVS --entrypoint /candidate/llama-server $IMG \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 \
  --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots -lv 4 \
  --mixed-batch-size "$CAP"
