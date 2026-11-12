#!/bin/bash
# prof.sh <tag> <lib> <depth> [extra env...]  nvprof llama-bench decode (-n 32) with full prod env (run-35b-gdn4-20261112.sh) + DEBUG; lock held externally
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D1; mkdir -p $R/prof
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=d1-nvprof; tag=$1 LIB=$2 d=$3; shift 3
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E=(); for a in GGML_CUDA_MOE_DOWNX=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_OCCAMY_FA_PIPE=3 GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_DEBUG=1 "$@"; do E+=(-e "$a"); done
printf '%s\n' "$@" > $R/prof/$tag.env.txt; sha256sum $LIB >> $R/prof/$tag.env.txt
docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $LIB:/app/libggml-cuda.so:ro -v $R/prof:/out -e LD_LIBRARY_PATH=/app "${E[@]}" \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/$tag.csv \
  /app/llama bench -m /models/occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -b 4096 -ub 4096 -p 0 -n 32 -d $d -r 1 > $R/prof/$tag.bench.txt 2>&1
echo "$tag rc=$?"; grep -E "occamy-gdn: (arepack|repack variant|arepacked)|Xid|error" $R/prof/$tag.bench.txt | head -4; grep -E "tg32" $R/prof/$tag.bench.txt | tail -1
