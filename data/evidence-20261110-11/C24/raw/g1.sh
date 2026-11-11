#!/bin/bash
J=/home/avg-usr/.claude/jobs/c24; W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; L=$W/cand/C24/build/bin/libggml-cuda.so
ON="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2"
for i in 1 2 3; do
  echo -n "A$i "; $J/g.sh C24 $L g1-A$i 64 2048
  echo -n "B$i "; $J/g.sh C24 $L g1-B$i 64 2048 $ON
done
for i in 2 3; do $J/cmp.sh g1-A1 g1-A$i; done
for i in 1 2 3; do $J/cmp.sh g1-A1 g1-B$i; done
echo G1-finished
