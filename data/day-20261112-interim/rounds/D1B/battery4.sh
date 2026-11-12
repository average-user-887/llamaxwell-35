#!/bin/bash
# (d) restore row-independence with identical KV layout: both seqs restored as in ed, seq0 teacher-forced on a different stream (S0:500000)
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1B; LIB=$M/occamy35-night-20261111/cand/D1B/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy, abort"; exit 1; }
cd $R
XENV=GGML_CUDA_OCCAMY_DEBUG=1 ./x.sh $LIB edx_pair1 pair 196608 1 L0:/slots/s0_98k_off0.bin S0:500000 L1:/slots/s1_90k_off200000.bin S1:290000 D32
./x.sh $LIB edx_pair0 pair 196608 0 L0:/slots/s0_98k_off0.bin S0:500000 L1:/slots/s1_90k_off200000.bin S1:290000 D32
python3 - <<'PY'
import numpy as np
def L(d,k): return np.fromfile(f'dumps/{d}/s{k}.logits',np.float32)
def st(d,k): return open(f'dumps/{d}/s{k}.state','rb').read()
for n in '10':
  a,b=f'ed_pair{n}',f'edx_pair{n}'; x,y=L(a,1),L(b,1); print(f'{a}~{b} s1: logits {"BITEQ" if np.array_equal(x,y) else "DIFF maxabs %.3g"%np.abs(x-y).max()} state {"BITEQ" if st(a,1)==st(b,1) else "DIFF"}')
PY
grep "N2SPLIT gather" dumps/edx_pair1/log.txt | sort | uniq -c | head -3
echo "done $(date +%T)"
