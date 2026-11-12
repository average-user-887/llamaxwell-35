#!/bin/bash
# prof.sh <tag> <depth> [extra env...]  nvprof one llama-bench decode (-n 6) at depth; lock held externally
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GDN1; mkdir -p $R/prof
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=gdn1-nvprof; tag=$1 d=$2; shift 2
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MOE_DOWNX=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 "$@"; do E+=(-e "$a"); done
docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $N/cand/GDN1/build/bin/libggml-cuda.so:/app/libggml-cuda.so:ro -v $R/prof:/out -e LD_LIBRARY_PATH=/app "${E[@]}" \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/$tag.csv \
  /app/llama bench -m /models/occamy-1.0-Q4_0-headq40b.gguf -ngl 999 -fa on -b 2048 -ub 1024 -p 0 -n 6 -d $d -r 1 > $R/prof/$tag.bench.txt 2>&1
echo "$tag rc=$?"; grep -E "tg6" $R/prof/$tag.bench.txt | tail -1
python3 -I /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/E07/ofa.py $R/prof/$tag.csv | tee $R/prof/$tag.ofa.txt
