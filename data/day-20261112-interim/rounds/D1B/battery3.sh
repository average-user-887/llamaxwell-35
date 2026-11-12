#!/bin/bash
# (d) restore + (b) cancel row-independence with depth-matched seq0 (FA split depends on batch max n_kv, so seq0 depth must match)
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1B; LIB=$M/occamy35-night-20261111/cand/D1B/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy, abort"; exit 1; }
cd $R
for n in 1 0; do
./x.sh $LIB edw_pair$n pair 196608 $n P0:300000:398027 L1:/slots/s1_90k_off200000.bin S1:290000 D32
./x.sh $LIB ebv_pair$n pair 32768 $n P0:50000:53000 P1:200000:203000 D16 X0 D16
done
./x.sh $LIB eb_pair0 pair 32768 0 P0:0:3000 P1:200000:203000 D16 X0 D16
python3 - <<'PY'
import numpy as np
def L(d,k): return np.fromfile(f'dumps/{d}/s{k}.logits',np.float32)
def st(d,k): return open(f'dumps/{d}/s{k}.state','rb').read()
def eq(a,b,k): x,y=L(a,k),L(b,k); return f'{a}~{b} s{k}: logits {"BITEQ" if np.array_equal(x,y) else "DIFF maxabs %.3g"%np.abs(x-y).max()} state {"BITEQ" if st(a,k)==st(b,k) else "DIFF"}'
for n in '10': print(eq(f'ed_pair{n}',f'edw_pair{n}',1)); print(eq(f'eb_pair{n}',f'ebv_pair{n}',1))
PY
echo "done $(date +%T)"
