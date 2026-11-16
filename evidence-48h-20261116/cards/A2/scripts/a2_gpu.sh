#!/usr/bin/env bash
# A2 moe_v3 shexp-idle-wave GPU checks (NOT run by the coder; the orchestrator queues it). ~20 min GPU hold.
# (1) microbench tools/moe3_mb (kernel SASS identical to the A2 lib): bitwise base moe_v3<true,false> vs
#     moe_v3<true,true> over 4 synthetic Q4_0 layers x 8 h vectors, then ABBA timing; run twice.
# (2) N3 short replay ABBA x1, both arms = A2 libs + bundle9 env (R0FLAGS); B adds GGML_CUDA_MOE_V3_SHIDLE=1,
#     first B also GGML_CUDA_MOE_V3_CHECK=64 (server-level bitwise vs moe_v2 and vs base moe_v3, expected 0 / 0).
# Env (launch only): N3_GPU, N3_UNLOAD_URL. POST unload only. Does not restore prod.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_MOE_V3_SHIDLE GGML_CUDA_MOE_V3_CHECK GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/A2; D=$N2/28h/h
IMG=local/m6000-build-tools:20261013
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/mb; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^na2-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
# (1) microbench
for k in 1 2; do
  timeout 300 docker run --rm --name "na2-mb$k-$$" --device "nvidia.com/gpu=$N3_GPU" -v "$E/tools:/t:ro" --entrypoint /t/moe3_mb "$IMG" 12 192 4 > $OUT/mb/run$k.log 2>&1
  S "mb run$k rc=$? $(grep -h 'BITWISE\|speedup' $OUT/mb/run$k.log | sed 's/.*speedup/speedup/;s/BITWISE base vs shidle: //' | tr '\n' '|')"
done
# (2) N3 short replay ABBA x1
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1 GGML_CUDA_OFA_RAGGED=1 GGML_CUDA_OFA_RAGGED_RATIO=2 GGML_CUDA_OCCAMY_GDN18=1 GGML_CUDA_GDN2_A2ROW=1"
i=0
for a in A B B A; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n; LIBS=$E/build/bin
  if [ $a = A ]; then X="$BASE"; elif [ $i = 2 ]; then X="$BASE GGML_CUDA_MOE_V3_SHIDLE=1 GGML_CUDA_MOE_V3_CHECK=64"; else X="$BASE GGML_CUDA_MOE_V3_SHIDLE=1"; fi
  export N3_EXTRA_ENV="$X" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$?"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  S "$n engaged: $(grep -h 'MOE_V3_SHIDLE launches\|MOE_V3_SHIDLE variant\|MOE_V3 launches' $L | tail -2 | tr '\n' '|')"
  [ $i = 2 ] && { grep -h 'MOE_V3 CHECK' $L > $OUT/check.log; S "CHECK lines $(wc -l < $OUT/check.log), nonzero: $(grep -vc 'bitwise-diff 0 / 2048, max_abs 0.000e+00; vs v3 bitwise-diff 0 / 2048' $OUT/check.log)"; }
done
S done
