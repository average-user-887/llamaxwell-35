#!/bin/bash
# SK1 phase 1 (run under flock gpu-bench.lock): G1 off/off/sf1, nvprof d0 off/sf1
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/SK1; L=$R/lib/t1.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
echo "lock acquired $(date)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps on M6000: $apps"
if [ "$apps" != 0 ]; then nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv; echo ABORT_GPU_BUSY; exit 3; fi
echo -n "A1 "; $R/g.sh SK1 $L g1-A1 64 2048
echo -n "B1 "; $R/g.sh SK1 $L g1-B1 64 2048 GGML_CUDA_OCCAMY_SMALLFUSE=1 GGML_CUDA_OCCAMY_DEBUG=1
$R/cmp.sh g1-A1 g1-B1
for s in step64 step2112; do echo "A1 vs AF1-A1 $s: $(sha256sum $R/dumps/g1-A1.$s.state|cut -c1-12)"; done
grep -c "occamy-sf1" $R/g1-B1.txt; grep -m2 "occamy-sf1" $R/g1-B1.txt
LIB=$L $R/prof.sh off-d0 0
LIB=$L $R/prof.sh on-d0 0 GGML_CUDA_OCCAMY_SMALLFUSE=1
python3 -I $R/busy.py $R/prof/off-d0.csv $R/prof/on-d0.csv
echo "PHASE1 DONE $(date)"
# phase 2 (target 2, t2 lib; flag-off SASS identical to t1)
L2=$R/lib/t2.so
echo -n "B2 "; $R/g.sh SK1 $L2 g1-B2 64 2048 GGML_CUDA_OCCAMY_SMALLFUSE=3 GGML_CUDA_OCCAMY_DEBUG=1
$R/cmp.sh g1-A1 g1-B2
echo "sf1 $(grep -c occamy-sf1 $R/g1-B2.txt) sf2 $(grep -c occamy-sf2 $R/g1-B2.txt)"; grep -m2 "occamy-sf2" $R/g1-B2.txt
LIB=$L2 $R/prof.sh on3-d0 0 GGML_CUDA_OCCAMY_SMALLFUSE=3
LIB=$L2 $R/prof.sh on2-d0 0 GGML_CUDA_OCCAMY_SMALLFUSE=2
python3 -I $R/busy.py $R/prof/off-d0.csv $R/prof/on-d0.csv $R/prof/on2-d0.csv $R/prof/on3-d0.csv
echo "PHASE2 DONE $(date)"
