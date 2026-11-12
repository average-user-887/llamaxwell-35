#!/bin/bash
# D1BF speed: ABBA x3 cycles, same D1B lib + prod env, N2SPLIT=1 both, A=N2GDN=0 B=N2GDN=1. (a) two slots restored deep (98k+90k) batch-2 decode,
# (b) single slot 98k, (c) warm_decode llama-bench. dxt skips first 8 steps. VRAM sampled by nvidia-smi during (a).
M=/media/avg-usr/Storage/m6000-llama-cpp; R=$M/occamy35-day-20261112/rounds/D1BF; LIB=$M/occamy35-night-20261111/cand/D1BF/build/bin/libggml-cuda.so
exec 9>$M/gpu-bench.lock; flock 9; echo "lock $(date +%T)"; curl -s -m 20 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c GPU-684f5e72); [ "$apps" = 0 ] || { echo "M6000 busy, abort"; exit 1; }
cd $R; rm -f $R/vram_runs.txt; export BIN=dxt XENV=DXQ=1
nvidia-smi -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-gpu=timestamp,memory.used,power.draw --format=csv,noheader,nounits -lms 500 > $R/vram_speed.csv & SMI=$!
for c in 1 2 3; do for a in A B B A; do n=$([ $a = A ] && echo 0 || echo 1)
 t0=$(date +%s.%N); ./x.sh $LIB sp_two_c${c}_$a pair 196608 $n L0:/slots/s0_98k_off0.bin S0:98000 L1:/slots/s1_90k_off200000.bin S1:290000 D104; echo "$a $t0 $(date +%s.%N)" >> $R/vram_runs.txt; done; done
kill $SMI
unset BIN XENV
# (b)/(c) skipped per advisor: n=1 kernels SASS-identical to the N2SPLIT build (sass/ compare)
python3 - <<'PY'
import re,glob,statistics as st
for k,f in (('two','pair_ms_mean'),):
    print(f'== {k} ({f}, ms/step, lower better)')
    ds=[]
    for c in (1,2,3):
        v={}
        for a in 'AB':
            xs=[float(re.search(f+r'=([\d.]+)',open(p).read()).group(1)) for p in sorted(glob.glob(f'dumps/sp_{k}_c{c}_{a}/log.txt'))]
            v[a]=st.mean(xs)
        d=100*(v['B']-v['A'])/v['A']; ds.append(d); print(f' cycle{c} A={v["A"]:.3f} B={v["B"]:.3f} d={d:+.2f}%')
    m=st.mean(ds); h=4.303*st.stdev(ds)/3**.5; print(f' mean d={m:+.2f}% 95%CI [{m-h:+.2f},{m+h:+.2f}] (t, n=3 cycles)')
mem=[int(l.split(',')[1]) for l in open('vram_speed.csv') if l.count(',')==2]; pw=[float(l.split(',')[2]) for l in open('vram_speed.csv') if l.count(',')==2]
import datetime
rows=[(datetime.datetime.strptime(l.split(',')[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp(),int(l.split(',')[1])) for l in open('vram_speed.csv') if l.count(',')==2]
pk={'A':[],'B':[]}
for l in open('vram_runs.txt'):
    a,t0,t1=l.split(); pk[a].append(max([m for t,m in rows if float(t0)<=t<=float(t1)] or [0]))
print('peak VRAM per run A',pk['A'],'B',pk['B'])
print('peak VRAM MiB',max(mem),'margin',24576-max(mem),'max W',max(pw))
PY
echo "done $(date +%T)"
