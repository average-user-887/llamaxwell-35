#!/bin/bash
M=<LAB>; G=$M/occamy35-day-20261112/evidence/G1c
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo busy; exit 90; }
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
export G1C_GPU=<gpu> G1C_NR=2 G1C_BUCKETS=32768 G1C_PER=1
curl -s -m60 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
sleep 5
[ "$(nvidia-smi -i $G1C_GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo apps; exit 91; }
OUT=$G/smoke-$(date -u +%H%M%S); mkdir -p $OUT; echo $OUT
for a in A B; do python3 -B $G/g1c_harness.py $a $OUT 0; done
docker ps -a --format '{{.Names}}' | grep '^g1c-' | xargs -r docker rm -f
