#!/bin/bash
# usage: g.sh <cand> <lib> <name> <steps> <long> [extra env KEY=VAL...]   (G1/G3 dxlogits run, lock held externally)
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/GDN4
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04
c=$1 lib=$2 nm=$3 st=$4 lg=$5; shift 5
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MOE_DOWNX=1 "$@"; do E+=(-e "$a"); done
mkdir -p $R/dumps
docker run --rm --name gdn4-g1 --network none -u $(id -u):$(id -g) --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $R/dumps:/o \
 -v $W/cand/$c-g1/g1app:/app:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" --entrypoint /app/dxlogits $IMG \
 /models/occamy-1.0-Q4_0-headq40b.gguf /o/$nm.bin $st /o/$nm $lg > $R/$nm.txt 2>&1; echo "rc=$?" >> $R/$nm.txt; grep -E "DONE|STATE|rc=" $R/$nm.txt | tr '\n' ' '; echo
