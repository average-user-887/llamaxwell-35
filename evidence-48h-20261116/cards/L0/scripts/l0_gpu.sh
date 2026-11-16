#!/usr/bin/env bash
# L0 N3 wall ledger: N3 short-profile replay x3 on the ledger build (d4c13eb + LLAMA_N3_LEDGER instrumentation, l0-ledger), prod batch8 env
# (PREFILL_SINGLE_STREAM, OCCAMY_DEBUG, cap1024, ub2048, b2048, 2x98K), ledger flag on. libggml-cuda.so is byte-identical to prod-libs/batch8-20261115.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER_SYNC
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/L0; D=$N2/28h/h
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nl0-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
LIBS=$E/build/bin
for i in 1 2 3; do
  n=L_$i; mkdir -p $OUT/$n/ledger; chmod 777 $OUT/$n/ledger
  export N3_EXTRA_ENV="$BASE" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 N3_LEDGER_DIR=$OUT/$n/ledger
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $E/n3_replay_l0.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$?"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  S "$n ledger files: $(ls $OUT/$n/ledger | tr '\n' ' ')"
done
python3 -I $E/summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
