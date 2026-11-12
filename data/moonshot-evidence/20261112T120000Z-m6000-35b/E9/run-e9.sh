#!/usr/bin/env bash
# E9 part 1 (timed). Env: E9_GPU, E9_UNLOAD_URL (POST .../api/models/unload/qwen36-35b-moe). Optional ORDER (default ABBA,ABBA), N_UNITS per position.
set -u
M=<LAB>; D=$M/occamy35-day-20261112/evidence/E9
: "${E9_GPU:?}" "${E9_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo busy; exit 2; }
OUT=$D/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
curl -s -m60 -X POST "$E9_UNLOAD_URL" > $OUT/unload.json
n=$(nvidia-smi -i "$E9_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "compute apps on GPU: $n; abort"; exit 4; }
python3 -I $D/e9_run.py $OUT "${ORDER:-ABBA,ABBA}" > $OUT/run.log 2>&1; rc=$?
docker ps -a --format '{{.Names}}' | grep '^e9-' | xargs -r docker rm -f >/dev/null
echo "rc=$rc out=$OUT"
