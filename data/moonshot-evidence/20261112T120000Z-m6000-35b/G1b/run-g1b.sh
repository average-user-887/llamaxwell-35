#!/usr/bin/env bash
# G1b GPU run. Env required: G1B_GPU (device id/uuid), G1B_UNLOAD_URL (full POST URL .../api/models/unload/qwen36-35b-moe). Optional ORDER.
set -u
M=<LAB>; G=$M/occamy35-day-20261112/evidence/G1b
: "${G1B_GPU:?}" "${G1B_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo "gpu-bench.lock busy"; exit 2; }
[ -x $G/build/bin/llama-server ] || { echo "no build"; exit 3; }
OUT=$G/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
curl -s -m60 -X POST "$G1B_UNLOAD_URL" > $OUT/unload.json
n=$(nvidia-smi -i "$G1B_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "GPU has $n compute apps; abort"; exit 4; }
nvidia-smi -i "$G1B_GPU" --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv -l 1 > $OUT/telemetry.csv & TP=$!
export G1B_GPU
python3 -I $G/g1b_run.py $OUT "${ORDER:-ABBA,BAAB,ABBA,BAAB,ABBA}" "$@" > $OUT/run.log 2>&1; rc=$?
kill $TP 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^g1b-' | xargs -r docker rm -f >/dev/null
echo "rc=$rc out=$OUT"
