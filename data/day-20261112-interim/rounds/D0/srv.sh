#!/bin/bash
# srv.sh <name-tag> [prof]: own test server d0-srv with prod GDN4 lib+env+launch shape (-np 2 -c 196608 -b 2048 -ub 1024), port 127.0.0.1:18490. lock held externally.
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D0; N=$M/occamy35-night-20261111
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; tag=$1; prof=$2; LIB=${LIB:-$M/prod-libs/libggml-cuda-gdn4-20261112.so}; RS=${RS:-$M/run-35b-gdn4-20261112.sh}
E=(); for a in $(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $RS | sed 's/-e //'); do E+=(-e "$a"); done
EP=(--entrypoint /app/llama-server); PRE=()
if [ "$prof" = prof ]; then EP=(--entrypoint /usr/local/cuda/bin/nvprof); PRE=(--csv --print-gpu-trace --log-file /out/$tag.csv /app/llama-server); fi
docker run -d --name d0-srv -p 127.0.0.1:18490:8080 --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $LIB:/app/libggml-cuda.so:ro -v $R:/out -v $R/slots:/slots -e LD_LIBRARY_PATH=/app "${E[@]}" "${EP[@]}" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  "${PRE[@]}" -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 -fa on --host 0.0.0.0 --port 8080 --metrics -lv 4 --no-warmup --slot-save-path /slots --cache-ram 0
