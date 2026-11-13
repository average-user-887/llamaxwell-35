#!/usr/bin/env bash
set -u
M=<LAB>
D=$M/occamy35-night-20261113/STOCK_B12318
export STOCK_GPU=<gpu>
UNLOAD_URL=http://<lab-endpoint>/api/models/unload/qwen36-35b-moe
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM

cleanup() {
  docker ps -a --format '{{.Names}}' | grep '^stock-b12318-' | xargs -r docker rm -f >/dev/null 2>&1
}
trap cleanup EXIT

exec 9>"$M/gpu-bench.lock"
flock -n 9 || { echo "gpu-bench.lock busy"; exit 2; }

curl -s -m60 -X POST "$UNLOAD_URL" > "$D/unload.json"
n=$(nvidia-smi -i "$STOCK_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "GPU has $n compute apps; abort"; exit 4; }

python3 -I "$D/bench_stock.py" "$D" > "$D/run.log" 2>&1
rc=$?
echo "rc=$rc out=$D"
exit $rc
