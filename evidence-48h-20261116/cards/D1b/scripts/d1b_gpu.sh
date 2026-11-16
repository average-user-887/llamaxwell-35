#!/usr/bin/env bash
# D1b: CUDA graphs on Maxwell, N3 A/B. arm prod = prod libs/env (GGML_CUDA_MAXWELL_GRAPHS absent) vs arm gr = same + GGML_CUDA_MAXWELL_GRAPHS=1 (TEST containers nd2-* only; owner OK 16 Nov, prod untouched).
# Both arms mount the D1b counter lib (libggml-cuda.so from branch d1b-graphcount, only change = GGML_CUDA_D1B_COUNT counters) over the prod batch8 libs. ABBA-ish order n=6 each, N3 short profile (4 turns x 1536 tok/slot, greedy).
# Stops at once if arm gr fails: CRASH.txt = sanitized failing stderr lines. Peak VRAM sampled every 5 s. ~45 min wall.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).  --plan prints the order and exits (no GPU, no lock).
set -u
ORD="prod,gr;gr,prod;gr,prod;prod,gr;prod,gr;gr,prod"
if [ "${1:-}" = "--plan" ]; then echo "$ORD" | tr ';' '\n' | nl -v0 -s': '; exit 0; fi
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_OFA_RAGGED GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER LLAMA_N3_LEDGER_SYNC LLAMA_N3_SCHED LLAMA_H1_CKPT
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/D1b; PRODLIBS=$M/prod-libs/batch8-20261115; LIB=$E/build/bin/libggml-cuda.so
[ "$(df --output=avail -BG $M | tail -1 | tr -dc 0-9)" -ge 15 ] || { echo "disk <15G"; exit 3; }
[ "$(sha256sum $LIB | cut -c1-64)" = 2b9c9a4010f20bcea1df92e09ba5b6eec07f9f8a88f478c508de352b35b8dd51 ] || { echo "counter lib sha mismatch"; exit 3; }
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
SAMP=""
cleanup() { [ -n "$SAMP" ] && kill $SAMP 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^nd2-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
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
export N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1 GGML_CUDA_D1B_COUNT=1" N3_CAND_DIR=$PRODLIBS N3B_GEMM_LIB=$LIB N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="$ORD" python3 -I $E/n3_replay_d1b.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; RC=$?; S "replay rc=$RC"
kill $SAMP 2>/dev/null; SAMP=""
python3 -I $E/summarize_d1b.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
[ -f $OUT/CRASH.txt ] && S "arm gr crashed: CRASH.txt"
S done
