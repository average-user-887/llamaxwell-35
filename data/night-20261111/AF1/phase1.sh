#!/bin/bash
# AF1 phase 1 (run under flock gpu-bench.lock): G1 off/off/on/on(mode2), nvprof d0 off/on
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/AF1; L=$N/cand/AF1/build/bin/libggml-cuda.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps on M6000: $apps"
if [ "$apps" != 0 ]; then nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv; echo ABORT_GPU_BUSY; exit 3; fi
echo -n "A1 "; $R/g.sh AF1 $L g1-A1 64 2048
echo -n "A2 "; $R/g.sh AF1 $L g1-A2 64 2048
echo -n "B1 "; $R/g.sh AF1 $L g1-B1 64 2048 GGML_CUDA_OCCAMY_ATTN_QKV=1
echo -n "B2 "; $R/g.sh AF1 $L g1-B2 64 2048 GGML_CUDA_OCCAMY_ATTN_QKV=2
for b in A2 B1 B2; do $R/cmp.sh g1-A1 g1-$b; done
grep -h "occamy-attn-qkv" $R/g1-B1.txt $R/g1-B2.txt
$R/prof.sh off-d0 0
$R/prof.sh on-d0 0 GGML_CUDA_OCCAMY_ATTN_QKV=1 GGML_CUDA_OCCAMY_DEBUG=1
grep -h "occamy-attn-qkv" $R/prof/on-d0.bench.txt
python3 -I $R/busy.py $R/prof/off-d0.csv $R/prof/on-d0.csv
echo "PHASE1 DONE $(date)"
