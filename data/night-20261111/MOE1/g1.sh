#!/bin/bash
# MOE1 G1: OFF (prod flags) vs ON (+GGML_CUDA_MOE_LOP3=1), same MOE1 lib. Lock held externally.
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/MOE1; L=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/cand/MOE1/build/bin/libggml-cuda.so
OFF="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2"
for i in 1 2; do echo -n "A$i "; $R/g.sh C24 $L g1-A$i 64 2048 $OFF; done
for i in 1 2; do echo -n "B$i "; $R/g.sh C24 $L g1-B$i 64 2048 $OFF GGML_CUDA_MOE_LOP3=1 GGML_CUDA_OCCAMY_DEBUG=1; done
$R/cmp.sh g1-A1 g1-A2; for i in 1 2; do $R/cmp.sh g1-A1 g1-B$i; done
grep -h "MOE_LOP3 launches" $R/g1-B1.txt | tail -1
echo G1-finished
