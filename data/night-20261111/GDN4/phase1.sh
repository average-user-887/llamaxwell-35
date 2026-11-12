#!/bin/bash
# GDN4 phase 1 (G1 + kill screen). Holds gpu-bench.lock; unloads the M6000 35B via per-model POST only.
N=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111; R=$N/rounds/GDN4; L=$R/libggml-cuda.gdn4.so; BL=$N/rounds/BUNDLE2/libggml-cuda.bundle2.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; SWAP_URL=http://172.21.0.1:11445
BASE="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_FA_PIPE=3"
ON="GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2"
exec 9>/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
echo "running before: $(curl -s -m10 $SWAP_URL/running)"
curl -s -m60 -X POST $SWAP_URL/api/models/unload/qwen36-35b-moe; echo
echo "running after: $(curl -s -m10 $SWAP_URL/running)"
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "apps $apps"; nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader
[ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
echo -n "A1 "; $R/g.sh C24 $L g1-A1 64 2048 $BASE
echo -n "B1 "; $R/g.sh C24 $L g1-B1 64 2048 $BASE $ON GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_DEBUG=1
$R/cmp.sh g1-A1 g1-B1
echo "repacked layers $(grep -c 'repacked ssm_out' $R/g1-B1.txt); nonzero-mismatch lines $(grep 'repacked ssm_out' $R/g1-B1.txt | grep -vc 'unpack mismatches 0')"
grep -E 'repack variant|blocks/SM -> grid|repack launches|pref launches' $R/g1-B1.txt | sort | uniq -c | head -12
grep -q "verdict=BITEQUAL" $R/g1-B1.cmp.txt && grep -q "A=652476ce477f B=652476ce477f" $R/g1-B1.cmp.txt && grep -q "A=5be5fc2d816d B=5be5fc2d816d" $R/g1-B1.cmp.txt || { echo G1-MISMATCH-STOP; exit 4; }
echo G1-PASS
for t in off1:$BL: on1:$L:GGML_CUDA_OCCAMY_GDN_REPACK=1 off2:$BL: on2:$L:GGML_CUDA_OCCAMY_GDN_REPACK=1 g64:$L:GGML_CUDA_OCCAMY_GDN_REPACK=1,GGML_CUDA_OCCAMY_GDN_RP_GRID=64; do
  IFS=: read tag lib ex <<< "$t"; LIB=$lib $R/prof.sh $tag 0 $ON ${ex//,/ }
done
python3 -I $R/period.py $R/prof/{off1,on1,off2,on2,g64}.csv
python3 -I $R/busy.py $R/prof/{off1,on1,off2,on2,g64}.csv
python3 -I $R/kstat.py $R/prof/{off1,on1,off2,on2,g64}.csv
grep -h 'repack variant' $R/prof/*.bench.txt | sort | uniq -c
echo "PHASE1 DONE $(date +%T)"
