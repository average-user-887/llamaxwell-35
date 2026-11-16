#!/usr/bin/env bash
# F6 leave-one-out of R0 bundle9 (X=all, Y=minus RAGGED, Z=minus GDN18; palindromic blocks x3). Derived from R0/gpu_r0.sh. Orig: N3 ABBA x3 (+ optional cap2048+gap arm C). A = batch8 libs (prod-libs/batch8-20261115, read-only), B = R0 bundle libs
# + GGML_CUDA_GDN2_A2ROW=1. Both: batch8 env (from run-35b-batch8-20261115.sh) + PREFILL_SINGLE_STREAM + OCCAMY_DEBUG, cap1024 ub2048 b2048.
# Env (launch only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/R0; F=$M/occamy35-48h-20261115/F6; D=$N2/28h/h
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$F/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $F/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nf6-' | xargs -r docker rm -f >/dev/null 2>&1; [ -n "${PRE:-x}" ] && docker ps -a --format '{{.Names}}' | grep '^n28h-' | grep -vxFf $OUT/pre_containers.txt | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"; docker ps -a --format '{{.Names}}' | grep '^n28h-' > $OUT/pre_containers.txt; true
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
RAG="GGML_CUDA_OFA_RAGGED=1 GGML_CUDA_OFA_RAGGED_RATIO=2"; G18="GGML_CUDA_OCCAMY_GDN18=1"; A2="GGML_CUDA_GDN2_A2ROW=1"
LIBS=$E/build/bin
i=0
for a in X Y Z Z Y X X Y Z Z Y X X Y Z Z Y X; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n
  case $a in X) F_="$RAG $G18 $A2";; Y) F_="$G18 $A2";; Z) F_="$RAG $A2";; esac
  export N3_EXTRA_ENV="$BASE $F_" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$?"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  rg=$(grep -h 'occamy FA ragged guard:' $L | wc -l); rr=$(grep -h 'occamy FA ragged guard:.*-> ragged' $L | wc -l); rf=$(grep -h 'ragged guard fallbacks=' $L | tail -1 | sed 's/.*fallbacks=//')
  g18=$(grep -h 'gdn18 engaged launches' $L | tail -1 | sed 's/.*launches //')
  V=ok
  case $a in X) { [ "${rg:-0}" -gt 0 ] && [ "${g18:-0}" -gt 0 ]; } || V=VOID;; Y) [ "${g18:-0}" -gt 0 ] || V=VOID; [ "${rg:-0}" -eq 0 ] || V=VOID;; Z) [ "${rg:-0}" -gt 0 ] || V=VOID; [ -z "$g18" ] || V=VOID;; esac
  S "$n arm=$a ragged_guard_lines=$rg to_ragged=$rr fallbacks=${rf:-none} gdn18_launches=${g18:-none} verdict=$V"
done
S done
