#!/bin/bash
# MOE1 nvprof metrics on moe_v2 (prod lib c9825b1), decode only d0. Run under flock on gpu-bench.lock.
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/MOE1; S=/media/avg-usr/Storage/m6000-llama-cpp
LIB=${LIB:-$S/prod-libs/libggml-cuda-call35-20261111.so}; TAG=${TAG:-prod}; EXTRA=${EXTRA:-}
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; CN=moe1-prof
trap "docker rm -f $CN >/dev/null 2>&1" EXIT
E="-e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 -e GGML_CUDA_MOE_DOWNX=1 -e GGML_CUDA_OCCAMY_GDN_QKREUSE=1 -e GGML_CUDA_OCCAMY_GDN_PCSHARE=1 -e GGML_CUDA_MOE_DOWNH=2 $EXTRA"
M=/models/occamy-1.0-Q4_0-headq40b.gguf
MET=${MET:-issue_slot_utilization,ipc,achieved_occupancy,dram_read_throughput,l2_read_throughput,shared_load_throughput,shared_store_throughput,stall_exec_dependency,stall_memory_dependency,stall_pipe_busy,stall_sync,stall_memory_throttle,stall_not_selected,stall_inst_fetch,stall_other,stall_texture,inst_executed,dram_utilization,shared_utilization}
docker run --rm --name $CN --network none --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $N/app:/app:ro -v $LIB:/app/libggml-cuda.so:ro -v $R/prof:/out -e LD_LIBRARY_PATH=/app $E \
  --entrypoint /usr/local/cuda/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --csv --kernels moe_v2 --metrics $MET --log-file /out/met-$TAG.csv \
  /app/llama bench -m $M -ngl 999 -fa on -b 4096 -ub 4096 -p 0 -n 2 -d 0 -r 1 > $R/prof/met-$TAG.bench.txt 2>&1
echo "rc=$?"
