#!/bin/bash
# D1B exactness battery (holds gpu-bench.lock). Same lib for all arms; solo = per-seq single-token decodes, pair = batch-2.
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1B; LIB=$M/occamy35-night-20261111/cand/D1B/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
curl -s -m 20 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy ($apps apps), abort"; exit 1; }
cd $R
E1="P0:0:3000 P1:200000:203000 D40"
EA="P1:200000:203000 D8 P0:0:3000 D32"
EB="P0:0:3000 P1:200000:203000 D16 X0 D16"
EC="P0:0:3000 P1:200000:206000 D12 X0 P0:100000:102000 D24"
EF="P0:0:3000 D4 C D16"
ED="L0:/slots/s0_98k_off0.bin S0:98000 L1:/slots/s1_90k_off200000.bin S1:290000 D32"
./x.sh $LIB e1_solo solo 32768 1 $E1; ./x.sh $LIB e1_pair1 pair 32768 1 $E1; ./x.sh $LIB e1_pair0 pair 32768 0 $E1
./x.sh $LIB ea_solo solo 32768 1 $EA; XENV=GGML_CUDA_OCCAMY_DEBUG=1 ./x.sh $LIB ea_pair1 pair 32768 1 $EA
./x.sh $LIB eb_solo solo 32768 1 $EB; ./x.sh $LIB eb_pair1 pair 32768 1 $EB
./x.sh $LIB ec_solo solo 32768 1 $EC; ./x.sh $LIB ec_pair1 pair 32768 1 $EC
./x.sh $LIB ef_solo solo 32768 1 $EF; XENV=GGML_CUDA_OCCAMY_DEBUG=1 ./x.sh $LIB ef_pair1 pair 32768 1 $EF
./x.sh $LIB ed_solo solo 196608 1 $ED; ./x.sh $LIB ed_pair1 pair 196608 1 $ED; ./x.sh $LIB ed_pair0 pair 196608 0 $ED
for c in e1_pair1 e1_pair0 ea_pair1 eb_pair1 ec_pair1 ef_pair1 ed_pair1 ed_pair0; do s=${c%_pair*}_solo; echo "== $c vs $s"; ./cmp2.sh $s $c; done
echo "done $(date +%T)"
