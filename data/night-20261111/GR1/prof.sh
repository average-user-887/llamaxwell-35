#!/bin/bash
# GR1 nvprof d0 OFF vs ON, lock held externally
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GR1
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=gr1-prof
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 -e GGML_CUDA_MOE_DOWNX=1 -e GGML_CUDA_OCCAMY_GDN_QKREUSE=1 -e GGML_CUDA_OCCAMY_GDN_PCSHARE=1 -e GGML_CUDA_MOE_DOWNH=2 -e GGML_CUDA_OCCAMY_FA_PIPE=3"
M=/models/occamy-1.0-Q4_0-headq40b.gguf
for arm in off on; do
 X=""; [ $arm = on ] && X="-e GGML_CUDA_MAXWELL_GRAPHS=1"
 docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $N/cand/E07/build/bin/libggml-cuda.so:/app/libggml-cuda.so:ro -v $R/prof-out:/out -e LD_LIBRARY_PATH=/app $E $X \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/d0-$arm.csv \
  /app/llama bench -m $M -ngl 999 -fa on -b 4096 -ub 4096 -p 0 -n 24 -d 0 -r 1 > $R/prof-out/d0-$arm.bench.txt 2>&1
 echo "$arm rc=$?"
done
