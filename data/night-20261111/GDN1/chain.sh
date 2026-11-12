#!/bin/bash
# GDN1 chain (run under flock gpu-bench.lock): GPU-free check, nvprof d0 (3L OFF, fused OFF, fused PREF), G1 OFF x2 / PREF x2 + cmp, G4 1-cycle sanity (ABBA x5 only if nvprof delta >= ~0.12 ms/tok, separate run)
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GDN1; L=$N/cand/GDN1/build/bin/libggml-cuda.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps on M6000: $apps"
if [ "$apps" != 0 ]; then nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv; echo ABORT_GPU_BUSY; exit 3; fi
P=GGML_CUDA_OCCAMY_FA_PIPE=3
$R/prof.sh off-d0 0 $P
$R/prof.sh pref-d0 0 $P GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_DEBUG=1
$R/prof.sh off3l-d0 0 $P GGML_CUDA_OCCAMY_GDN_3L=1
ON="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 $P"
echo -n "A1 "; $R/g.sh GDN1 $L g1-A1 64 2048 $ON
echo -n "A2 "; $R/g.sh GDN1 $L g1-A2 64 2048 $ON
echo -n "B1 "; $R/g.sh GDN1 $L g1-B1 64 2048 $ON GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_DEBUG=1
echo -n "B2 "; $R/g.sh GDN1 $L g1-B2 64 2048 $ON GGML_CUDA_OCCAMY_GDN_PREF=1
for b in A2 B1 B2; do $R/cmp.sh g1-A1 g1-$b; done
grep -h "occamy-gdn" $R/g1-B1.txt | head -5
bash $R/g4.sh $R/cfg-gdn1sanity.json warm_decode
echo "CHAIN DONE $(date)"
