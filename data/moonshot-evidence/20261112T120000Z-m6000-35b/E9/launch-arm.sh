#!/usr/bin/env bash
# E9 server launcher. Usage: launch-arm.sh <name-suffix> <lv:default|4> [extra docker opts...]  -> container e9-<suffix>, 127.0.0.1:18494
# Called only by run-e9.sh / run-e9-gdb.sh under gpu-bench.lock. GPU id from env E9_GPU. G1b bundle3 build binaries (needed: prod-image server cannot read these slot files) + prod launcher env verbatim + prod cuda lib (ro) over the build's. Single slot, -c 98304 = prod per-slot context.
set -euo pipefail
NAME=e9-$1; LV=$2; shift 2
M=<LAB>; E=$M/occamy35-day-20261112/evidence
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
: "${E9_GPU:?}"
ENVS=$(grep -o -e '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-bundle3-20261112.sh | tr '\n' ' ')
echo "$ENVS" | grep -q -E 'MAXWELL_GRAPHS|MAXWELL_GEMM' && { echo forbidden; exit 3; }
[ -z "${GGML_CUDA_MAXWELL_GRAPHS:-}${GGML_CUDA_FA_MAXWELL_GEMM:-}" ] || { echo forbidden; exit 3; }
LVARG=""; [ "$LV" = default ] || LVARG="-lv $LV"
docker run -d --name "$NAME" -p 127.0.0.1:18494:8080 --device nvidia.com/gpu=$E9_GPU "$@" \
  -v <MODELS>:/models:ro -v $E/E9/slots:/slots \
  -v $E/G1b/build/bin:/candidate:ro -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/candidate/libggml-cuda.so:ro \
  -e LD_LIBRARY_PATH=/candidate $ENVS --entrypoint /candidate/llama-server $IMG \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 1 -c 98304 -b 2048 -ub 1024 \
  --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots $LVARG
