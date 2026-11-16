#!/usr/bin/env bash
# B4 multirow decode FA microbench (short GPU hold, < 10 min): RMSE vs FP64 (fa_rmse14, nq = k rows, causal in block,
# 2 streams, gap 777) and test-backend-ops eval + perf on the mr_causal cases. Arms: stock k>1 (MULTIROW unset) vs multirow R=auto/1/2/4.
# Env (launch only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OFA_MULTIROW GGML_CUDA_OFA_MR_R
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/B4; G=$M/occamy35-night-20261114/14/gates
IMG=local/m6000-build-tools:20261013
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/rmse $OUT/tbo; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nb4-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 900 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; abort"; exit 4; }
BASE=(-e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FA_PIPE=3 -e GGML_CUDA_OCCAMY_FA_ADDR=1 -e GGML_CUDA_OCCAMY_DEBUG=1)
run() { # tag timeout envs... -- cmd...
  local tag=$1 to=$2; shift 2; local ev=(); while [ "$1" != "--" ]; do ev+=(-e "$1"); shift; done; shift
  timeout $to docker run --rm --name "nb4-$tag-$$" --device "nvidia.com/gpu=$N3_GPU" -v "$E/build/bin:/b:ro" -v "$G:/g:ro" -w /b \
    -e LD_LIBRARY_PATH=/b "${BASE[@]}" "${ev[@]}" "$IMG" "$@"
}
# (1) RMSE vs FP64, kv 90112, 2 streams
for k in 1 3 8 16; do
  run stock_k$k 120 FA_NS=2 X=1 -- /g/fa_rmse14 90112 $k 1 > $OUT/rmse/stock_k$k.log 2>&1; S "rmse stock k$k rc=$?"
  [ $k = 1 ] && continue
  for R in 0 1 2 4; do
    run mr_k${k}_R$R 120 FA_NS=2 GGML_CUDA_OFA_MULTIROW=1 GGML_CUDA_OFA_MR_R=$R -- /g/fa_rmse14 90112 $k 1 > $OUT/rmse/mr_k${k}_R$R.log 2>&1; S "rmse mr k$k R$R rc=$?"
  done
done
# (2) test-backend-ops eval (vs CPU) on mr_causal cases, multirow on
run eval_mr 240 GGML_CUDA_OFA_MULTIROW=1 -- /b/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 -p mask=mr_causal > $OUT/tbo/eval_mr.log 2>&1; S "eval mr rc=$?"
# (3) perf, 90K depth, k = 1..16
run perf_stock 120 X=1 -- /b/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p mask=mr_causal > $OUT/tbo/perf_stock.log 2>&1; S "perf stock rc=$?"
for R in 0 1 2 4; do
  run perf_mr_R$R 120 GGML_CUDA_OFA_MULTIROW=1 GGML_CUDA_OFA_MR_R=$R -- /b/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p mask=mr_causal > $OUT/tbo/perf_mr_R$R.log 2>&1; S "perf mr R$R rc=$?"
done
run perf_mr_R0b 120 GGML_CUDA_OFA_MULTIROW=1 -- /b/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p mask=mr_causal > $OUT/tbo/perf_mr_R0b.log 2>&1; S "perf mr R0 repeat rc=$?"
S "done"
