#!/bin/bash
# MOE2 kill screen: nvprof gpu-trace d0, A=LOP3, B=+TAILBAL, order A B B A. Run under flock on gpu-bench.lock.
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/MOE2; L=$N/cand/MOE1/build/bin/libggml-cuda.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=moe2-prof
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 -e GGML_CUDA_MOE_DOWNX=1 -e GGML_CUDA_OCCAMY_GDN_QKREUSE=1 -e GGML_CUDA_OCCAMY_GDN_PCSHARE=1 -e GGML_CUDA_MOE_DOWNH=2 -e GGML_CUDA_MOE_LOP3=1"
M=/models/occamy-1.0-Q4_0-headq40b.gguf; mkdir -p $R/prof
for t in A1 B1 B2 A2; do
 X=""; [[ $t == B* ]] && X="-e GGML_CUDA_MOE_TAILBAL=1"
 docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $L:/app/libggml-cuda.so:ro -v $R/prof:/out -e LD_LIBRARY_PATH=/app $E $X \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --print-gpu-trace --log-file /out/$t.csv \
  /app/llama bench -m $M -ngl 999 -fa on -b 4096 -ub 4096 -p 0 -n 6 -d 0 -r 1 > $R/prof/$t.bench.txt 2>&1
 echo "$t rc=$?"
done
for t in A1 B1 B2 A2; do echo -n "$t "; python3 -I $R/an.py $R/prof/$t.csv; done
