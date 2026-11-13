#!/bin/bash
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
cd <LAB>/occamy35-night-20261113/N2
exec 9><LAB>/gpu-bench.lock
flock -w 7200 9 || exit 90
S=$(date +%s); echo "locked2 $(date -u +%T)" >> run/steps.log
curl -s -m60 -X POST <lab-endpoint>/api/models/unload/qwen36-35b-moe > run/unload2.out 2>&1
GPU=<gpu>
for i in $(seq 1 60); do [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo busy >> run/steps.log; exit 91; }
G1C_DIR=<LAB>/occamy35-night-20261113/NC2/bin G1C_GPU=$GPU G1C_NR=64 G1C_PER=2 G1C_CUT_EPOCH=$((S+4200)) python3 -B g1c_harness.py Ap <LAB>/occamy35-night-20261113/N2/run 0,1,2,3,4,5,6,7 > run/run2.log 2>&1
echo "harness2 rc=$? $(date -u +%T)" >> run/steps.log
docker ps -a --format '{{.Names}}' | grep '^n2-' | xargs -r docker rm -f >/dev/null
