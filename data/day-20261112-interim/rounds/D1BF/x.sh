#!/bin/bash
# x.sh <lib> <name> <mode> <n_ctx> <n2gdn 0|1 (N2SPLIT=1 always)> <events...>   (gpu lock held by caller)
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1BF; N=$M/occamy35-night-20261111; D0=$M/occamy35-day-20261112/rounds/D0
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; RS=$M/run-35b-gdn4-20261112.sh
lib=$1 nm=$2 mode=$3 nc=$4 sp=$5; shift 5
E=(); for a in $(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $RS | sed 's/-e //'); do E+=(-e "$a"); done
E+=(-e GGML_CUDA_OCCAMY_N2SPLIT=1 -e GGML_CUDA_OCCAMY_N2GDN=$sp); for a in $XENV; do E+=(-e "$a"); done
mkdir -p $R/dumps/$nm
if [ -n "$PROF" ]; then EP=(--entrypoint /usr/local/cuda/bin/nvprof $IMG --print-gpu-trace --csv --log-file /o/prof.csv /app2/dx2seq); else EP=(--entrypoint /app2/dx2seq $IMG); fi
docker run --rm --name d1bf-x --network none -u $(id -u):$(id -g) --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $R/dumps/$nm:/o \
 -v $D0/deep_text.txt:/t.txt:ro -v $D0/slots:/slots:ro -v $N/app:/app:ro -v $R/scripts/${BIN:-dx2seq}:/app2/dx2seq:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" -w /app "${EP[@]}" \
 /models/occamy-1.0-Q4_0-headq40b.gguf /t.txt /o $mode $nc "$@" > $R/dumps/$nm/log.txt 2>&1; echo "rc=$?" >> $R/dumps/$nm/log.txt
echo "$nm: $(grep -E 'defaults|^rc=|DONE|failed|STATE' $R/dumps/$nm/log.txt | tr '\n' ' ')"
