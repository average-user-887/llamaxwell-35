#!/usr/bin/env bash
# FA0: prefill flash-attn split + launch/tile sweep at the 84K-start two-slot N3 short replay (prod batch8 env, fa0-split build).
# Each config = one N3 short replay (np 2, start 84K, 4 turns); GGML_CUDA_FA_SPLIT gives per-phase ms for keys>=81920 ops.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only), N3_PORT, N3_HOST (replay driver endpoint). Optional FA0_CONFIGS="name:ENV=V,ENV=V;..." to override.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}" "${N3_HOST:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_FA GGML_CUDA_OCCAMY_KVQ8 GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER_SYNC
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/FA0; D=$N2/28h/h
export N3_GPU N3_PORT N3_HOST PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nfa-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1 GGML_CUDA_FA_SPLIT=/n3l/fa0.tsv"
LIBS=$E/build/bin
# A first and last = noise floor; one existing knob changed per row
DEF="A:;conv128:GGML_CUDA_FA_GEMM_CONV_BLOCK=128;conv512:GGML_CUDA_FA_GEMM_CONV_BLOCK=512;convgrid2048:GGML_CUDA_FA_GEMM_CONV_GRID=2048;tileq32:GGML_CUDA_FA_GEMM_TILEQ=32;splitk2048:GGML_CUDA_FA_GEMM_SPLITK=2048;splitk8192:GGML_CUDA_FA_GEMM_SPLITK=8192;chunkq512:GGML_CUDA_FA_GEMM_CHUNKQ=512;fused:GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1;fused_splitk2048:GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1 GGML_CUDA_FA_GEMM_SPLITK=2048;A2:"
CFG="${FA0_CONFIGS:-$DEF}"
IFS=';' read -ra ROWS <<< "$CFG"
for row in "${ROWS[@]}"; do
  n=${row%%:*}; ev=${row#*:}; T=$OUT/T_$n; mkdir -p $T/nt; chmod 777 $T/nt
  export N3_EXTRA_ENV="$BASE $ev" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 N3_NT_DIR=$T/nt
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $E/n3_replay_fa0.py $T --no-sweep --no-ctxfull > $T/run.log 2>&1; S "$n rc=$?"
  S "$n files: $(ls $T/nt | tr '\n' ' ')"
done
python3 -I $E/fa0_summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
