#!/bin/bash
# GDN3 card B: G1 (all 30 layers repacked, DEBUG host unpack check) + nvprof pilot (1 layer) at grids 72/64/48. Holds the GPU lock.
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GDN3; A=$R/libggml-cuda.cardA.so; B=$R/libggml-cuda.cardB.so
PE="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_FA_PIPE=3"
exec 9>/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
nvidia-smi --id=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-gpu=memory.used,memory.total --format=csv,noheader
$R/g.sh x $A g1-A1 64 2048 $PE
$R/g.sh x $B g1-B1 64 2048 $PE GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_DEBUG=1
grep -E "repack|blocks/SM" $R/g1-B1.txt | sort | uniq -c | head -8
$R/cmp.sh g1-A1 g1-B1
for g in 72 64 48; do LIB=$B $R/prof.sh rpB-g$g 0 GGML_CUDA_OCCAMY_FA_PIPE=3 GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_GDN_REPACK_LAYERS=1 GGML_CUDA_OCCAMY_GDN_RP_GRID=$g >/dev/null; grep -E "repack" $R/prof/rpB-g$g.bench.txt | head -3; grep tg6 $R/prof/rpB-g$g.bench.txt | tail -1; done
echo "unlock $(date +%T)"
