#!/bin/bash
M=/media/avg-usr/Storage/m6000-llama-cpp; W=$M/occamy35-day-20261112/rounds/B3; P=$M/prod-libs/libggml-cuda-gdn4-20261112.so; L=$W/libggml-cuda.b3.so
GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e; SWAP_URL=http://<lab-endpoint>
PENV=$(grep -o -- '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-gdn4-20261112.sh | sed 's/-e //' | tr '\n' ' ')
BEXTRA="GGML_CUDA_OCCAMY_N2SPLIT=1 GGML_CUDA_OCCAMY_FA_ADDR=1 GGML_CUDA_OCCAMY_N2GDN=1"
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
curl -s -m60 -X POST $SWAP_URL/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "apps $apps $(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader)"
[ "$apps" != 0 ] && { echo ABORT_GPU_BUSY; exit 3; }
t0=$(date +%s)
echo -n "G1 A "; $W/g.sh $P g1-A $PENV
echo -n "G1 B "; $W/g.sh $L g1-B $PENV $BEXTRA
$W/cmp.sh g1-A g1-B .step64.state .step2112.state
grep -q "verdict=BITEQUAL" $W/g1-B.cmp.txt && [ "$(grep -c 'diffbytes=0' $W/g1-B.cmp.txt)" = 2 ] || { echo G1-FAIL-STOP; exit 5; }
echo "G1-PASS $(( $(date +%s)-t0 ))s $(date +%T)"
(cd /tmp; python3 -I $W/b3_deep.py ABBA 4 64) > $W/b3_deep.log 2>&1; echo "deep rc=$? $(date +%T)"; tail -12 $W/b3_deep.log
(cd /tmp; python3 -I $W/b3_deep.py ABBA1 4 64) > $W/b3_deep1.log 2>&1; echo "deep1 rc=$? $(date +%T)"; tail -8 $W/b3_deep1.log
for a in B A; do
  nvidia-smi -i $GPU --query-gpu=timestamp,memory.used --format=csv,noheader -lms 500 > $W/smoke_vram$([ $a = B ] && echo "" || echo "_A").csv & SP=$!
  cases=abcde; [ $a = A ] && cases=a
  (cd /tmp; python3 -I $W/smoke.py $a $cases) > $W/smoke_$a.out 2>&1; echo "smoke $a rc=$? $(date +%T)"; cat $W/smoke_$a.out
  kill $SP
done
$W/g4.sh $W/cfg-b3.json warm_decode
docker rm -f b3-srv b3-g1 b3-smoke >/dev/null 2>&1
echo "ALL DONE $(date +%T) $(( $(date +%s)-t0 ))s"
