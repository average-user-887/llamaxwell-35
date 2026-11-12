#!/bin/bash
# gd.sh <lib> <name> [env...]   deep bitexact: prefill 16384 of D0 deep_text, 64 teacher-forced decode steps at pos 16384.. (lock held externally)
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D1ATT; D0=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D0
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; lib=$1 nm=$2; shift 2
E=(); for a in "$@"; do E+=(-e "$a"); done
docker run --rm --name att-gd --network none -u $(id -u):$(id -g) --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $R/dumps:/o -v $D0/deep_text.txt:/t.txt:ro \
 -v $W/cand/C24-g1/g1app:/app:ro -v $R/scripts/dxdeep:/app2/dxdeep:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" -w /app --entrypoint /app2/dxdeep $IMG \
 /models/occamy-1.0-Q4_0-headq40b.gguf /t.txt /o/$nm.bin /o/$nm.state 16384 64 20480 > $R/$nm.txt 2>&1; echo "rc=$?" >> $R/$nm.txt; grep -E "n_vocab|PREFILL|DONE|STATE|rc=|addr mode|pipe mode" $R/$nm.txt | sort -u | tr '\n' ' '; echo
