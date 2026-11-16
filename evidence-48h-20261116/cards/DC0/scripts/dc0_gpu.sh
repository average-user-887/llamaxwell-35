#!/usr/bin/env bash
# DC0: decode scaling + launch census. One N3 short replay (np 2, prod batch8 env) on the dc0 build with GGML_CUDA_NODE_TIMING + SPLIT: 1-row (decode) vs 2-row (dec2) per class. ~1 h max (timing arm is slower than prod).
# Output: run-*/summary.md via dc0_summarize.py. No GGML_CUDA_MAXWELL_GRAPHS. Never run GET unload-all.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only), N3_PORT, N3_HOST (replay driver endpoint).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}" "${N3_HOST:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_FA GGML_CUDA_OCCAMY_KVQ8 GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER_SYNC
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/DC0; D=$N2/28h/h
export N3_GPU N3_PORT N3_HOST PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^ndc-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1 GGML_CUDA_NODE_TIMING_SPLIT=1"
LIBS=$E/build/bin
n=T_1; T=$OUT/$n; mkdir -p $T/nt; chmod 777 $T/nt
export N3_EXTRA_ENV="$BASE" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 N3_NT_DIR=$T/nt
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" timeout 3300 python3 -I $E/n3_replay_dc0.py $T --no-sweep --no-ctxfull > $T/run.log 2>&1; S "$n rc=$?"
S "$n files: $(ls $T/nt | tr '\n' ' ')"
python3 -I $E/dc0_summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
