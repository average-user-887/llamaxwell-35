#!/usr/bin/env bash
# Card 30: host flag A/B. 9/bin all arms, env GEMM FUSED SINGLE_STREAM MOE_V3 DEBUG. Order A B C D D C B A. Env: N3_GPU, N3_UNLOAD_URL (launch only).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_MOE_V3 LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; N2=$M/occamy35-night-20261114; E=$N2/30; D=$E/h
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^n30-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 14400 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1 LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1"; export N3_CAND_DIR=$N2/9/bin N3B_GEMM_LIB=$N2/9/bin/libggml-cuda.so
i=0
for a in A B C D D C B A; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n
  N3_EXTRA_ENV="$BASE"; N3_BATCH=2048; N3_CKPT=16; N3_CKPT_STEP=2048; CAP=1024; UB=2048; N3_SERVER_ARGS=""
  case $a in B) N3_SERVER_ARGS="--threads 4 --threads-batch 4 --poll 100";; C) N3_SERVER_ARGS="--poll 0";; D) N3_SERVER_ARGS="--prio 2 --poll 100";; esac
  export N3_SERVER_ARGS N3_EXTRA_ENV N3_BATCH N3_CKPT N3_CKPT_STEP
  FULL_UB=$UB N3_CAP=$CAP N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$? cap=$CAP ub=$UB b=$N3_BATCH args=[$N3_SERVER_ARGS]"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  g=$(grep -h 'occamy_fa_gemm: engaged' $L | tail -1 | sed 's/.*engaged=\([0-9]*\).*/\1/'); f=$(grep -h 'occamy_fa_gemm_fused: calls' $L | tail -1 | sed 's/.*calls=\([0-9]*\).*/\1/'); m=$(grep -h 'MOE_V3 launches' $L | tail -1 | sed 's/.*launches //')
  S "$n engaged gemm=${g:-0} fused=${f:-0} moe_v3=${m:-0}"
  S "$n flagcheck: $(grep -hi 'invalid argument\|unknown argument\|error: ' $L | head -2 | tr '\n' '|')"
  S "$n ss_lines: $(grep -hi 'single.stream\|single_stream' $L | head -3 | tr '\n' '|')"
  S "$n n_ubatch: $(grep -h -o 'n_batch *= *[0-9]*\|n_ubatch *= *[0-9]*' $L | head -2 | tr '\n' ' ') oom: $(grep -hci 'out of memory\|failed to allocate\|cudaMalloc failed' $L | head -1)"
done
S done
python3 -I $E/summ30.py $OUT $E/RESULT.md >> $OUT/steps.log 2>&1
