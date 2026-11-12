#!/usr/bin/env bash
# E9 part 2 (untimed): "gdb sampling, not perf". Env: E9_GPU, E9_UNLOAD_URL. Needs E9/slots/e9_np1_98k.bin from part 1 (or it is prefilled here).
# DEADLINE (epoch s) optional env stops sampling.
set -u
M=<LAB>; D=$M/occamy35-day-20261112/evidence/E9
: "${E9_GPU:?}" "${E9_UNLOAD_URL:?}"; NS=${NS:-100}
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo busy; exit 2; }
OUT=$D/gdb-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
curl -s -m60 -X POST "$E9_UNLOAD_URL" > $OUT/unload.json
[ "$(nvidia-smi -i "$E9_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] || { echo "compute apps; abort"; exit 4; }
trap 'docker rm -f e9-gdb >/dev/null 2>&1' EXIT
$D/launch-arm.sh gdb default --cap-add SYS_PTRACE >/dev/null || exit 5
for i in $(seq 300); do curl -s -m2 localhost:18494/health | grep -q '"ok"' && break; sleep 2; done
docker exec e9-gdb sh -c 'apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gdb >/dev/null 2>&1; gdb --version | head -1' > $OUT/gdb_install.txt 2>&1
python3 -I $D/e9_gdb_drive.py $OUT $NS > $OUT/drive.log 2>&1
echo "out=$OUT"
