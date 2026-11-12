#!/bin/bash
# G1 runner. G1_GPU and G1_UNLOAD_URL are supplied via environment (placeholders <lab-endpoint> in docs).
M=<LAB>; OUT=<LAB>/occamy35-day-20261112/evidence/G1/20261112T171109Z
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo "lock busy"; exit 90; }
echo "start $(date -u +%FT%TZ)" > $OUT/steps.log
curl -s -m60 -X POST "$G1_UNLOAD_URL" > $OUT/unload.out 2>&1; echo "unload rc=$?" >> $OUT/steps.log
n=$(nvidia-smi -i $G1_GPU --query-compute-apps=pid --format=csv,noheader | wc -l); echo "compute apps=$n" >> $OUT/steps.log
[ "$n" = 0 ] || { echo busy >> $OUT/steps.log; exit 91; }
run() { arm=$1; min=$2; shift 2; PYTHONDONTWRITEBYTECODE=1 python3 -B $OUT/g1_harness.py $arm $OUT $min "$@"; echo "arm $arm rc=$? $(date -u +%T)" >> $OUT/steps.log; }
# C: generic FP32 attention (fa off, F32 KV); try ub512 then ub256
run C 32 18 66000 -fa off -ctk f32 -ctv f32 -ub 512 || true
if ! ls $OUT/C/*_r0_65536.npy >/dev/null 2>&1; then mv $OUT/C $OUT/C_try1_failed; run C 32 18 66000 -fa off -ctk f32 -ctv f32 -ub 256; fi
run A 32 18 98560
run B 30 18 98560
run Ap 12 0 98560
run A2 12 0 98560
run B2 12 0 98560
echo "end $(date -u +%FT%TZ)" >> $OUT/steps.log
