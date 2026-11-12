#!/bin/bash
# Row-independence battery: stock batch-2 != solo even at N2SPLIT=0 (different n=2 kernels), so exactness gate = seq1 outputs
# bit-identical when seq0's content changes (any wrong-row gather/state mix would change them), plus stock pair0 references.
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1B; LIB=$M/occamy35-night-20261111/cand/D1B/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy ($apps apps), abort"; exit 1; }
cd $R
for n in 1 0; do
./x.sh $LIB e1v_pair$n pair 32768 $n P0:50000:53000 P1:200000:203000 D40
./x.sh $LIB eav_pair$n pair 32768 $n P1:200000:203000 D8 P0:60000:63000 D32
./x.sh $LIB ecv_pair$n pair 32768 $n P0:70000:73000 P1:200000:206000 D12 X0 P0:110000:112000 D24
./x.sh $LIB edv_pair$n pair 196608 $n P0:0:3000 L1:/slots/s1_90k_off200000.bin S1:290000 D32
done
./x.sh $LIB e1_pair0b pair 32768 0 P0:0:3000 P1:200000:203000 D40
./x.sh $LIB ea_pair0 pair 32768 0 P1:200000:203000 D8 P0:0:3000 D32
./x.sh $LIB ec_pair0 pair 32768 0 P0:0:3000 P1:200000:206000 D12 X0 P0:100000:102000 D24
python3 - <<'PY'
import numpy as np,re
def L(d,k): return np.fromfile(f'dumps/{d}/s{k}.logits',np.float32)
def st(d,k): return open(f'dumps/{d}/s{k}.state','rb').read()
def eq(a,b,k): x,y=L(a,k),L(b,k); return f'{a}~{b} s{k}: logits {"BITEQ" if np.array_equal(x,y) else "DIFF maxabs %.3g"%np.abs(x-y).max()} state {"BITEQ" if st(a,k)==st(b,k) else "DIFF"}'
for n in '10':
  print(eq(f'e1_pair{n}',f'e1v_pair{n}',1)); print(eq(f'ea_pair{n}',f'eav_pair{n}',1)); print(eq(f'ec_pair{n}',f'ecv_pair{n}',1))
print(eq('e1_pair0','e1_pair0b',0), '| run-to-run determinism')
nv=248320
def top(a,b,k):
  x,y=L(a,k),L(b,k); n=x.size//(y.size//y.size*1); x=x.reshape(-1,y.size//(y.size//x.size) if False else -1) if False else x
  return
PY
for c in ea_pair0 ec_pair0; do echo "== $c vs ${c%_pair0}_solo"; ./cmp2.sh ${c%_pair0}_solo $c; done
echo "== edv: seq1 (restored 90k) vs ed_pair (seq0 = 98k restore) per arm"; for n in 1 0; do ./cmp2.sh ed_pair$n edv_pair$n | sed -n '3,4p'; done
echo "done $(date +%T)"
