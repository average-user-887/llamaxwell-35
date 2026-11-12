#!/bin/bash
# g.sh <lib> <name> [env KEY=VAL...]   G1 dxlogits 64 steps + states step64/step2112 (lock held externally)
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/B3
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; IMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; lib=$1 nm=$2; shift 2
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MOE_DOWNX=1 "$@"; do E+=(-e "$a"); done
mkdir -p $R/dumps
docker run --rm --name b3-g1 --network none -u $(id -u):$(id -g) --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $R/dumps:/o \
 -v $W/cand/C24-g1/g1app:/app:ro -v $lib:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${E[@]}" --entrypoint /app/dxlogits $IMG \
 /models/occamy-1.0-Q4_0-headq40b.gguf /o/$nm.bin 64 /o/$nm 2048 > $R/$nm.txt 2>&1; echo "rc=$?" >> $R/$nm.txt; grep -E "DONE|STATE|rc=|addr mode|pipe mode" $R/$nm.txt | sort -u | tr '\n' ' '; echo
