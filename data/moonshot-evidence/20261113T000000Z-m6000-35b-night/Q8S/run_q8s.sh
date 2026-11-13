#!/bin/bash
# Q8S GPU run. Holds gpu-bench.lock for the whole run.
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
M=<LAB>; N=$M/occamy35-night-20261113; Q=$N/Q8S; L=$Q/long
export G1C_GPU=<gpu> Q8S_GPU=<gpu>
exec 9>$M/gpu-bench.lock; flock 9; T0=$(date +%s); S=$Q/steps.log; echo "lock $(date -u +%T)" >> $S
curl -s -m60 -X POST <lab-endpoint>/api/models/unload/qwen36-35b-moe > $Q/unload.out 2>&1; echo "unload rc=$?" >> $S
n=$(nvidia-smi -i $G1C_GPU --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { echo "busy $n; abort" >> $S; exit 91; }
export G1C_CUT_EPOCH=$((T0+70*60)) G1C_NR=64 G1C_PER=2
echo "== VRAM" >> $S; $Q/vram_q8.sh f16:f16:f16 q8kv:q8_0:q8_0 q8k:q8_0:f16 q8v:f16:q8_0 >> $Q/vram.log 2>&1; echo "vram done $(date -u +%T)" >> $S
for arm in Q8KV Q8K; do python3 -B $Q/g1c_harness_q8.py $arm $L 0,1,2,3 >> $Q/long.log 2>&1; echo "long $arm rc=$? $(date -u +%T)" >> $S; done
$Q/shallow_q8.sh Q8KV Q8K Q8V >> $Q/shallow.log 2>&1; echo "shallow done $(date -u +%T)" >> $S
python3 -B $Q/g1c_harness_q8.py Q8V $L 0,1,2,3 >> $Q/long.log 2>&1; echo "long Q8V rc=$? $(date -u +%T)" >> $S
docker ps -a --format '{{.Names}}' | grep '^nq8s-\|^g1c-q8' | xargs -r docker rm -f >/dev/null
python3 -I $Q/q8s_analyze.py $L > $Q/q8s_analyze.out 2>&1; python3 -I $Q/shallow_analyze.py $M/occamy35-day-20261112/rounds/D1B/dumps $Q/dumps > $Q/shallow_analyze.out 2>&1
echo "end $(date -u +%T)" >> $S
