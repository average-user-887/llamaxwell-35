#!/bin/bash
echo "lock acquired $(date)"; GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
echo -n "B4 "; /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/g.sh AF1 /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/cand/AF1/build/bin/libggml-cuda.so g1-B4 64 2048 GGML_CUDA_OCCAMY_ATTN_QKV=4
/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/cmp.sh g1-A1 g1-B4; grep -h occamy-attn-qkv /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/g1-B4.txt
/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh off2-d0 0
/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh m4-d0 0 GGML_CUDA_OCCAMY_ATTN_QKV=4
/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof.sh m3-d0 0 GGML_CUDA_OCCAMY_ATTN_QKV=3
python3 -I /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/busy.py /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/off2-d0.csv /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/m4-d0.csv /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/prof/m3-d0.csv
echo "PHASE1B DONE $(date)"
