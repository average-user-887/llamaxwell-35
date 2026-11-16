#!/usr/bin/env bash
# A5b N3 co-decode hold: LLAMA_N3_SCHED=0 vs 4 (arms p0, p4; p3 optional via A5B_P3=1) in ABBA order, n=6 each (A5B_BLOCKS=6 default), N3 short profile, prod batch8 env + A5b build
# (a5b-codecode off a5-sched; libggml-cuda.so and llama-server byte-identical to prod). Per-slot token ids saved. Peak VRAM sampled every 5 s. summarize.py -> summary.md with accept lines.
# Wall: ~3.5 min per session x 12 = ~45 min (x18 = ~65 min with p3).  Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).
# Optional overrides forwarded to the server: LLAMA_N3_SCHED_HOLD (ms, 150), LLAMA_N3_SCHED_HOLDB (max held batches per episode, 4), LLAMA_N3_SCHED_REACH (tokens, 4096), CHUNK/MAXDEFER for p3.
# --plan prints the session order and exits (no GPU, no lock).
set -u
ORD="p0,p4;p4,p0;p4,p0;p0,p4;p0,p4;p4,p0"
if [ "${A5B_P3:-0}" = 1 ]; then ORD="p0,p4,p3;p3,p4,p0;p4,p0,p3;p3,p0,p4;p0,p3,p4;p4,p3,p0"; fi
if [ "${A5B_BLOCKS:-6}" = 4 ]; then ORD=$(echo "$ORD" | cut -d';' -f1-4); fi
if [ "${1:-}" = "--plan" ]; then echo "$ORD" | tr ';' '\n' | nl -v0 -s': '; exit 0; fi
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_OFA_RAGGED GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER LLAMA_N3_SCHED
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/A5b; LIBS=$E/build/bin
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
SAMP=""
cleanup() { [ -n "$SAMP" ] && kill $SAMP 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^na5b-' | grep -v '^na5b-build' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
# VRAM sampler every 5 s: epoch,MiB (whole-GPU used; the lock guarantees nothing else runs)
( while :; do echo "$(date +%s.%N),$(nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" >> $OUT/vram.csv; sleep 5; done ) &
SAMP=$!
EX="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
for v in LLAMA_N3_SCHED_HOLD LLAMA_N3_SCHED_HOLDB LLAMA_N3_SCHED_REACH LLAMA_N3_SCHED_CHUNK LLAMA_N3_SCHED_MAXDEFER; do [ -n "${!v:-}" ] && EX="$EX $v=${!v}"; done
export N3_EXTRA_ENV="$EX" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="$ORD" python3 -I $E/n3_replay_a5b.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "replay rc=$?"
kill $SAMP 2>/dev/null; SAMP=""
python3 -I $E/summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
