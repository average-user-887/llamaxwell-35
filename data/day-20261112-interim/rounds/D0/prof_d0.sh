#!/bin/bash
# prof_d0.sh <tag> <depth> <n>: nvprof of llama bench with prod GDN4 lib + full prod env (parsed from run script), -b 2048 -ub 1024. lock held externally
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D0; N=$M/occamy35-night-20261111
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=d0-nvprof; tag=$1 d=$2 n=$3; LIB=${LIB:-$M/prod-libs/libggml-cuda-gdn4-20261112.so}; RS=${RS:-$M/run-35b-gdn4-20261112.sh}
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E=(); for a in $(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $RS | sed 's/-e //'); do E+=(-e "$a"); done
echo "env: ${E[*]}" > $R/$tag.env.txt
docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $LIB:/app/libggml-cuda.so:ro -v $R:/out -e LD_LIBRARY_PATH=/app "${E[@]}" \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/$tag.csv \
  /app/llama bench -m /models/occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -b 2048 -ub 1024 -p 0 -n $n -d $d -r 1 > $R/$tag.bench.txt 2>&1
echo "$tag rc=$?"; grep -E "tg" $R/$tag.bench.txt | tail -1
