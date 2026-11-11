#!/bin/bash
# E07 phase 1 (run under flock gpu-bench.lock): GPU-free check, nvprof d16384 pipe 0/1/2, G1 off/off/pipe1/pipe2
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/E07; L=$N/cand/E07/build/bin/libggml-cuda.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps on M6000: $apps"
if [ "$apps" != 0 ]; then nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv; echo ABORT_GPU_BUSY; exit 3; fi
ON="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2"
for p in 0 1 2; do $R/prof.sh p$p-d16384 16384 GGML_CUDA_OCCAMY_FA_PIPE=$p; done
echo -n "A1 "; $R/g.sh E07 $L g1-A1 64 2048 $ON
echo -n "A2 "; $R/g.sh E07 $L g1-A2 64 2048 $ON
echo -n "B1 "; $R/g.sh E07 $L g1-B1 64 2048 $ON GGML_CUDA_OCCAMY_FA_PIPE=1
echo -n "B2 "; $R/g.sh E07 $L g1-B2 64 2048 $ON GGML_CUDA_OCCAMY_FA_PIPE=2
for b in A2 B1 B2; do $R/cmp.sh g1-A1 g1-$b; done
grep -h "occamy FA" $R/g1-B1.txt $R/g1-B2.txt
echo "PHASE1 DONE $(date)"
