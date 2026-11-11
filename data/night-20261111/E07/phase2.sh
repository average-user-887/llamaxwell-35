#!/bin/bash
# E07 phase 2 (under flock): nvprof d16384 pipe 0/3/4, G1 A3(off, new lib)/B3/B4 vs A1
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/E07; L=$N/cand/E07/build/bin/libggml-cuda.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date) lib=$(sha256sum $L | cut -c1-16)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps on M6000: $apps"
if [ "$apps" != 0 ]; then echo ABORT_GPU_BUSY; exit 3; fi
ON="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2"
for p in 0 3 4; do $R/prof.sh h$p-d16384 16384 GGML_CUDA_OCCAMY_FA_PIPE=$p; done
echo -n "A3 "; $R/g.sh E07 $L g1-A3 64 2048 $ON
echo -n "B3 "; $R/g.sh E07 $L g1-B3 64 2048 $ON GGML_CUDA_OCCAMY_FA_PIPE=3
echo -n "B4 "; $R/g.sh E07 $L g1-B4 64 2048 $ON GGML_CUDA_OCCAMY_FA_PIPE=4
for b in A3 B3 B4; do $R/cmp.sh g1-A1 g1-$b; done
grep -h "occamy FA pipe" $R/g1-B3.txt $R/g1-B4.txt
echo "PHASE2 DONE $(date)"
