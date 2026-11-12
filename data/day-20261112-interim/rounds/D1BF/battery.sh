#!/bin/bash
# D1BF exactness: same lib, N2SPLIT=1 both arms, pair (batch-2) N2GDN=0 vs N2GDN=1; bitequal logits + states, both seqs.
# e1 2-slot base, ea reorder [1,0], eb cancel, ec reuse, ef clear(C), ed restore D0 98K/90K slots.
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1BF; LIB=$M/occamy35-night-20261111/cand/D1BF/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; echo "wait lock $(date +%T)"; flock 9; echo "lock $(date +%T)"
curl -s -m 20 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy ($apps apps), abort"; exit 1; }
cd $R
declare -A EV=( [e1]="P0:0:3000 P1:200000:203000 D40" [ea]="P1:200000:203000 D8 P0:0:3000 D32" [eb]="P0:0:3000 P1:200000:203000 D16 X0 D16"
 [ec]="P0:0:3000 P1:200000:206000 D12 X0 P0:100000:102000 D24" [ef]="P0:0:3000 D4 C D16" [ed]="L0:/slots/s0_98k_off0.bin S0:98000 L1:/slots/s1_90k_off200000.bin S1:290000 D32" )
for c in e1 ea eb ec ef ed; do nc=32768; [ $c = ed ] && nc=196608
 ./x.sh $LIB ${c}_g0 pair $nc 0 ${EV[$c]}; XENV=GGML_CUDA_OCCAMY_DEBUG=1 ./x.sh $LIB ${c}_g1 pair $nc 1 ${EV[$c]}
 echo "== $c g1 vs g0 ($(grep -o 'n2gdn launches[^,]*' dumps/${c}_g1/log.txt | tail -1))"; ./cmp2.sh ${c}_g0 ${c}_g1; done
dmesg 2>/dev/null | grep -i xid | tail -2
echo "done $(date +%T)"
