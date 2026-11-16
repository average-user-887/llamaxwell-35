#!/usr/bin/env bash
# R0 bundle9: N3 ABBA x3 (+ optional cap2048+gap arm C). A = batch8 libs (prod-libs/batch8-20261115, read-only), B = R0 bundle libs
# + GGML_CUDA_GDN2_A2ROW=1. Both: batch8 env (from run-35b-batch8-20261115.sh) + PREFILL_SINGLE_STREAM + OCCAMY_DEBUG, cap1024 ub2048 b2048.
# Env (launch only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/R0; D=$N2/28h/h
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nr0-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 14400 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
R0FLAGS="GGML_CUDA_OFA_RAGGED=1 GGML_CUDA_OFA_RAGGED_RATIO=2 GGML_CUDA_OCCAMY_GDN18=1 GGML_CUDA_GDN2_A2ROW=1"
i=0
for a in A B B A A B B A A B B A ${R0_ARM_C:+C C}; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n
  if [ $a = A ]; then LIBS=$M/prod-libs/batch8-20261115; X="$BASE"; elif [ $a = B ]; then LIBS=$E/build/bin; X="$BASE $R0FLAGS"; else LIBS=$E/build/bin; X="$BASE $R0FLAGS LLAMA_ARG_MIXED_BATCH_PREFILL_MAX=${PMAX:-1152}"; fi
  export N3_EXTRA_ENV="$X" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
  FULL_UB=2048 N3_CAP=$([ $a = C ] && echo 2048 || echo 1024) N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$? libs=$(basename $LIBS)"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  S "$n engaged: $(grep -h 'occamy FA ragged on\|ragged guard\|n2gdn variant\|gdn18 engaged\|gdn18 variant' $L | tail -4 | tr '\n' '|')"
  v=$(grep -hc 'n2gdn variant' $L | head -1); l=$(grep -h 'occamy-gdn: n2gdn launches' $L | tail -1 | sed 's/.*launches //'); m=$(grep -h 'MOE_V3 launches' $L | tail -1 | sed 's/.*launches //')
done
S done
