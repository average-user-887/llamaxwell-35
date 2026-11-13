#!/usr/bin/env bash
# N1 runner. Env: N1_GPU, N1_UNLOAD_URL. Usage: n1_run.sh <outdir> <arm A|B> ; holds gpu-bench.lock (taken by caller wrapper n1_all.sh)
set -u
M=<LAB>; E=$M/occamy35-day-20261112/evidence/G1b; OUT=$1; ARM=$2
CIMG=nvidia/cuda:12.8.1-devel-ubuntu24.04
NAME=n1-srv
ENVS=$(grep -o -e '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-bundle3-20261112.sh | tr '\n' ' ')
echo "$ENVS" | grep -q -E 'MAXWELL' && exit 3
[ -z "${GGML_CUDA_MAXWELL_GRAPHS:-}" ] || exit 3
LIB=$M/prod-libs/libggml-cuda-bundle3-20261112.so; XENV=""
if [ "$ARM" = B ]; then LIB=$M/codex-investigation-20261112/prefill/gemm/integration-v4/src/build/bin/libggml-cuda.so; XENV="-e GGML_CUDA_FA_MAXWELL_GEMM=1"; fi
n=$(nvidia-smi -i "$N1_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" -eq 0 ] || { echo "GPU busy"; exit 4; }
mkdir -p $OUT/arm$ARM; docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME -p 127.0.0.1:18494:8080 --device nvidia.com/gpu=$N1_GPU \
  -v <MODELS>:/models:ro -v $E/slots:/slots -v $OUT/arm$ARM:/out \
  -v $E/build/bin:/candidate:ro -v $LIB:/candidate/libggml-cuda.so:ro \
  -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib $ENVS $XENV --init --entrypoint /usr/local/cuda-12.8/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --print-gpu-trace --csv --log-file /out/trace.csv /candidate/llama-server \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 \
  --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots
for i in $(seq 1 120); do curl -s -m2 http://127.0.0.1:18494/health | grep -q ok && break; docker ps -q -f name=$NAME | grep -q . || { echo "server died"; docker logs $NAME | tail -20; exit 5; }; sleep 5; done
python3 -I $M/occamy35-night-20261113/N1/n1_drive.py $OUT/arm$ARM $ARM
docker exec $NAME sh -c 'kill -INT $(pidof llama-server)'; for i in $(seq 1 90); do docker ps -q -f name=$NAME | grep -q . || break; sleep 2; done
docker logs $NAME > $OUT/arm$ARM/server.log 2>&1; docker rm -f $NAME >/dev/null
