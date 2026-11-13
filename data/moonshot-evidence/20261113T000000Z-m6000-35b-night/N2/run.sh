#!/bin/bash
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
cd <LAB>/occamy35-night-20261113/N2; mkdir -p run shallow
exec 9><LAB>/gpu-bench.lock
flock -w 7200 9 || { echo lock timeout > run/steps.log; exit 90; }
S=$(date +%s); echo "locked $(date -u +%T)" > run/steps.log
curl -s -m60 -X POST <lab-endpoint>/api/models/unload/qwen36-35b-moe > run/unload.out 2>&1
export N2_GPU=<gpu>
for i in $(seq 1 60); do [ "$(nvidia-smi -i $N2_GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i $N2_GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "busy abort" >> run/steps.log; exit 91; }
echo "gpu idle $(date -u +%T)" >> run/steps.log
./shallow_n2.sh A2 Ap > shallow.log 2>&1; echo "shallow done $(date -u +%T)" >> run/steps.log
python3 -I shallow_an.py <LAB>/occamy35-day-20261112/rounds/D1B/dumps shallow > shallow_table.txt 2>&1
G1C_DIR=<LAB>/occamy35-night-20261113/NC2/bin G1C_GPU=$N2_GPU G1C_NR=64 G1C_PER=2 G1C_CUT_EPOCH=$((S+4500)) python3 -B g1c_harness.py Ap run 0,1,2,3,4,5,6,7 > run/run.log 2>&1
echo "harness rc=$? $(date -u +%T)" >> run/steps.log
docker ps -a --format '{{.Names}}' | grep '^n2-' | xargs -r docker rm -f >/dev/null
