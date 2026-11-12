#!/bin/bash
# D1ATT chained GPU run: G1 (once) + deep bitexact (once) + ABBA warm_decode/d4096 (3 cycles) + deep server ABBA on D0 slots. Holds gpu-bench.lock throughout.
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1ATT; L=$R/libggml-cuda.att1.so; P=$M/prod-libs/libggml-cuda-gdn4-20261112.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; SWAP_URL=http://<lab-endpoint>
BASE="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_FA_PIPE=3"
PENV=$(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-gdn4-20261112.sh | sed 's/-e //' | tr '\n' ' ')
now=$(date +%s); go=$(date -d 11:30 +%s); [ $now -lt $go ] && { echo "waiting for D0 priority window until 11:30"; sleep $((go-now)); }
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
echo "running before: $(curl -s -m10 $SWAP_URL/running)"; curl -s -m60 -X POST $SWAP_URL/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "apps $apps $(nvidia-smi -i $GPU --query-gpu=memory.used,power.draw --format=csv,noheader)"
[ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
t0=$(date +%s)
if [ -f $M/occamy35-day-20261112/rounds/D0/slots/s0_98k_off0.bin ] && [ -f $M/occamy35-day-20261112/rounds/D0/slots/s1_90k_off200000.bin ]; then
  (cd /tmp; python3 -I $R/scripts/att_deep.py ABBA 4 64) > $R/deep_abba.log 2>&1; echo "deep rc=$?"; tail -3 $R/deep_abba.log
else echo "NO D0 SLOT FILES, deep ABBA skipped"; fi
# G1 passed once 09:14 (run_all.1.log), not repeated
echo -n "DEEP A "; $R/gd.sh $L gd-A $PENV
echo -n "DEEP B "; $R/gd.sh $L gd-B $PENV GGML_CUDA_OCCAMY_FA_ADDR=1
$R/cmp.sh gd-A gd-B .state
grep -q "verdict=BITEQUAL" $R/gd-B.cmp.txt && grep -q "diffbytes=0" $R/gd-B.cmp.txt || { echo DEEP-MISMATCH-STOP; exit 5; }
echo "DEEP-PASS $(( $(date +%s)-t0 ))s"
nvidia-smi -i $GPU --query-gpu=clocks.sm,clocks.mem,power.limit --format=csv,noheader
$R/g4.sh $R/cfg-att1.json warm_decode
$R/g4.sh $R/cfg-att1.json d4096
echo "ABBA-bench done $(( $(date +%s)-t0 ))s"
docker rm -f att-srv att-g1 att-gd >/dev/null 2>&1
echo "running end: $(curl -s -m10 $SWAP_URL/running)"; echo "ALL DONE $(date +%T) $(( $(date +%s)-t0 ))s"
