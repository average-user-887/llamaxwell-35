#!/usr/bin/env bash
# H1 checkpoint-copy A/B: arm prod (H1 build, LLAMA_H1_CKPT unset = prod path) vs arm h1 (LLAMA_H1_CKPT=1) in ABBA order, n=6 each (H1_BLOCKS=6 default; 4 => first four rows),
# N3 short profile (4 turns x 1536 tokens per slot, greedy), prod batch8 env, ledger on in both arms (stages ckpt_create / ckpt_wait_gpu / ckpt_copy).
# Peak VRAM sampled every 5 s. summarize.py -> summary.md (makespan, ckpt_create sum per replay, token identity vs first prod session, peak VRAM, engagement).
# Wall: ~3.5 min per session x 12 = ~45 min.  Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).
# --plan prints the session order and exits (no GPU, no lock).
set -u
ORD="prod,h1;h1,prod;h1,prod;prod,h1;prod,h1;h1,prod"
if [ "${H1_BLOCKS:-6}" = 4 ]; then ORD=$(echo "$ORD" | cut -d';' -f1-4); fi
if [ "${1:-}" = "--plan" ]; then echo "$ORD" | tr ';' '\n' | nl -v0 -s': '; exit 0; fi
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_OFA_RAGGED GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER LLAMA_N3_LEDGER_SYNC LLAMA_N3_SCHED LLAMA_H1_CKPT
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/H1; LIBS=$E/build/bin
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/ledger; chmod 777 $OUT/ledger; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
SAMP=""
cleanup() { [ -n "$SAMP" ] && kill $SAMP 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^nh1-' | grep -v '^nh1-build' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
( while :; do echo "$(date +%s.%N),$(nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" >> $OUT/vram.csv; sleep 5; done ) &
SAMP=$!
export N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 N3_LEDGER_DIR=$OUT/ledger
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="$ORD" python3 -I $E/n3_replay_h1.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "replay rc=$?"
kill $SAMP 2>/dev/null; SAMP=""
python3 -I $E/summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
