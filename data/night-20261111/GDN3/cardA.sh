#!/bin/bash
# GDN3 card A (under flock gpu-bench.lock): nvprof d0, gdn_fused grid 48 vs 24 (fused + 3L), moe_v2 grid cap 24
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GDN3; GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps: $apps"; [ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
B="GGML_CUDA_OCCAMY_FA_PIPE=3 GGML_CUDA_OCCAMY_DEBUG=1"
$R/prof.sh g48 0 $B
$R/prof.sh g24 0 $B GGML_CUDA_OCCAMY_GDN_GRID=24
$R/prof.sh g48-3l 0 $B GGML_CUDA_OCCAMY_GDN_3L=1
$R/prof.sh g24-3l 0 $B GGML_CUDA_OCCAMY_GDN_3L=1 GGML_CUDA_OCCAMY_GDN_GRID=24
$R/prof.sh moe24 0 $B GGML_CUDA_OCCAMY_MOE_GRID=24
echo "CARDA DONE $(date)"
