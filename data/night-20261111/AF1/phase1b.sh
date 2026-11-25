#!/bin/bash
echo "lock acquired $(date)"; GPU=GPU-REDACTED
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
echo -n "B4 "; <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/g.sh AF1 <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/cand/AF1/build/bin/libggml-cuda.so g1-B4 64 2048 GGML_CUDA_OCCAMY_ATTN_QKV=4
<STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/cmp.sh g1-A1 g1-B4; grep -h occamy-attn-qkv <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/g1-B4.txt
<STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh off2-d0 0
<STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh m4-d0 0 GGML_CUDA_OCCAMY_ATTN_QKV=4
<STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh m3-d0 0 GGML_CUDA_OCCAMY_ATTN_QKV=3
python3 -I <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/busy.py <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/off2-d0.csv <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/m4-d0.csv <STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/m3-d0.csv
echo "PHASE1B DONE $(date)"
