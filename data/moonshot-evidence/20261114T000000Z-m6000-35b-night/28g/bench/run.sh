#!/usr/bin/env bash
# Card 28g: moe_v3 split microbench under the GPU lock. Env (launch command only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; D=$M/occamy35-night-20261114/28g/bench; X=${1:-mb}; R=_$X
exec 9>$M/gpu-bench.lock; flock -w 14400 9 || exit 90
echo "lock $(date +%T)" > $D/steps.log
curl -s -m60 -X POST "$N3_UNLOAD_URL" > /dev/null
for i in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo busy >> $D/steps.log; exit 4; }
docker run --rm --name n28g-$X --network none --device nvidia.com/gpu=$N3_GPU -v $D:/w:ro -e LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 /w/$X > $D/result$R.txt 2>&1
echo "done $(date +%T)" >> $D/steps.log
exec 9>&-
