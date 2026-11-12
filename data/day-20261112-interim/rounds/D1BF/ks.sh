#!/bin/bash
# D1BF kill screen: occupancy print (DEBUG run) + nvprof GPU trace of short pair decode, N2GDN=0 vs 1, same lib
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1BF; LIB=$M/occamy35-night-20261111/cand/D1BF/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; echo "wait lock $(date +%T)"; flock 9; echo "lock $(date +%T)"
curl -s -m 20 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy ($apps apps), abort"; exit 1; }
cd $R; E1="P0:0:3000 P1:200000:203000 D40"
XENV=GGML_CUDA_OCCAMY_DEBUG=1 ./x.sh $LIB ks_dbg1 pair 32768 1 $E1
grep -E "n2gdn|launches|Xid" dumps/ks_dbg1/log.txt | head
PROF=1 ./x.sh $LIB ks_p0 pair 32768 0 $E1
PROF=1 ./x.sh $LIB ks_p1 pair 32768 1 $E1
./cmp2.sh ks_p0 ks_p1
dmesg 2>/dev/null | grep -i xid | tail -2
echo "done $(date +%T)"
