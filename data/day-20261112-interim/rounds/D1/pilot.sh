#!/bin/bash
# D1 pilot under flock: unload swap models, occupancy via in-process API log (grid capped 72), then 96 only if 4 blocks/SM
D=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D1; B=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/GDN4/libggml-cuda.gdn4.so; C=$D/libggml-cuda.gdn5.so; SW=http://<lab-endpoint>
date; curl -s -m 10 $SW/running; echo
for m in $(curl -s -m 10 $SW/running | python3 -I -c 'import sys,json; print(" ".join(r.get("model","") for r in json.load(sys.stdin).get("running",[])))' 2>/dev/null); do echo unload $m; curl -s -m 60 -X POST $SW/api/models/unload/$m; echo; done
nvidia-smi --id=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-gpu=memory.used,power.limit,clocks.sm,temperature.gpu --format=csv
nvidia-smi --id=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-compute-apps=pid,used_memory --format=csv
x0=$(journalctl -k -q --since today 2>/dev/null | grep -c Xid)
$D/prof.sh base1 $B 0
$D/prof.sh a72 $C 0 GGML_CUDA_OCCAMY_GDN_AREPACK=1 GGML_CUDA_OCCAMY_GDN_AREPACK_LAYERS=1 GGML_CUDA_OCCAMY_GDN_RA_GRID=72
if grep -q 'arepack variant .* 4 blocks/SM' $D/prof/a72.bench.txt; then
  $D/prof.sh a96 $C 0 GGML_CUDA_OCCAMY_GDN_AREPACK=1 GGML_CUDA_OCCAMY_GDN_AREPACK_LAYERS=1
else echo "occupancy not 4/SM: no grid>72 run"; fi
$D/prof.sh base2 $B 0
$D/prof.sh a72b $C 0 GGML_CUDA_OCCAMY_GDN_AREPACK=1 GGML_CUDA_OCCAMY_GDN_AREPACK_LAYERS=1 GGML_CUDA_OCCAMY_GDN_RA_GRID=72
[ -f $D/prof/a96.csv ] && $D/prof.sh a96b $C 0 GGML_CUDA_OCCAMY_GDN_AREPACK=1 GGML_CUDA_OCCAMY_GDN_AREPACK_LAYERS=1
echo "Xid before $x0 after $(journalctl -k -q --since today 2>/dev/null | grep -c Xid)"
python3 -I $D/pstat.py $D/prof/*.csv
date
