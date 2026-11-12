#!/bin/bash
# BUNDLE1 G1: A=bundle lib flags OFF, B=bundle lib all ON, C=E07 lib. Lock held externally.
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/BUNDLE1
BL=$N/cand/BUNDLE/build/bin/libggml-cuda.so; EL=$N/cand/E07/build/bin/libggml-cuda.so
BASE="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_FA_PIPE=3"
ON="GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1"
echo "start $(date)"; nvidia-smi -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-compute-apps=pid --format=csv,noheader | wc -l
echo -n "A1 "; $R/g.sh C24 $BL g1-A1 64 2048 $BASE
echo -n "C1 "; $R/g.sh C24 $EL g1-C1 64 2048 $BASE
echo -n "B1 "; $R/g.sh C24 $BL g1-B1 64 2048 $BASE $ON GGML_CUDA_OCCAMY_DEBUG=1
echo -n "B2 "; $R/g.sh C24 $BL g1-B2 64 2048 $BASE $ON
for b in C1 B1 B2; do $R/cmp.sh g1-A1 g1-$b; done
grep -hiE "tailbal|hostgap|gdn.*pref|lop3" $R/g1-B1.txt | sort | uniq -c | sort -rn | head -12
echo G1-finished $(date)
