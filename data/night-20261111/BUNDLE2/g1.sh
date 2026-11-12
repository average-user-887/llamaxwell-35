#!/bin/bash
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/BUNDLE2
BL=$R/libggml-cuda.bundle2.so
BASE="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_FA_PIPE=3"
ON="GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2"
echo "start $(date)"; apps=$(nvidia-smi -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-compute-apps=pid --format=csv,noheader | wc -l); echo "apps $apps"; [ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
echo -n "A1 "; $R/g.sh C24 $BL g1-A1 64 2048 $BASE
echo -n "B1 "; $R/g.sh C24 $BL g1-B1 64 2048 $BASE $ON GGML_CUDA_OCCAMY_DEBUG=1
for b in B1; do $R/cmp.sh g1-A1 g1-$b; done
echo "sf2 count $(grep -c occamy-sf2 $R/g1-B1.txt)"; grep -m2 occamy-sf2 $R/g1-B1.txt
echo G1-finished $(date)
