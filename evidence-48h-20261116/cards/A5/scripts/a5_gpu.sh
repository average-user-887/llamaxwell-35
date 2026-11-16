#!/usr/bin/env bash
# A5 N3 slot-scheduling ABBA-style: policies LLAMA_N3_SCHED=0,1,2,3 (arms p0..p3 of n3_replay_a5.py) interleaved over 6 blocks (4 Williams rows + 2 extra rows = each policy 6x,
# position-balanced), N3 short profile, prod batch8 env + A5 build (a5-sched; libggml-cuda.so byte-identical to prod). Per-slot token ids saved; summarize.py -> summary.md.
# Wall estimate: ~3.5 min per session (L0 measured 3.3-3.5 min incl. start/restore) x 24 = ~85 min; policy 1 may lengthen it (deferral) -> allow 2 h. Set A5_BLOCKS=4 for a ~60 min run.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).  LLAMA_N3_SCHED_CHUNK / LLAMA_N3_SCHED_MAXDEFER may be exported to override defaults (512 / 64).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_OFA_RAGGED GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER LLAMA_N3_SCHED
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/A5; LIBS=$E/build/bin
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
ORD="p0,p1,p3,p2;p1,p2,p0,p3;p2,p3,p1,p0;p3,p0,p2,p1;p0,p2,p1,p3;p3,p1,p2,p0"
if [ "${A5_BLOCKS:-6}" = 4 ]; then ORD="p0,p1,p3,p2;p1,p2,p0,p3;p2,p3,p1,p0;p3,p0,p2,p1"; fi
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^na5-' | grep -v '^na5-build' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
# LLAMA_N3_SCHED_* are forwarded into the container via N3_EXTRA_ENV (policy itself is set per arm p0..p3 inside the harness)
EX="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
[ -n "${LLAMA_N3_SCHED_CHUNK:-}" ] && EX="$EX LLAMA_N3_SCHED_CHUNK=$LLAMA_N3_SCHED_CHUNK"
[ -n "${LLAMA_N3_SCHED_MAXDEFER:-}" ] && EX="$EX LLAMA_N3_SCHED_MAXDEFER=$LLAMA_N3_SCHED_MAXDEFER"
export N3_EXTRA_ENV="$EX" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="$ORD" python3 -I $E/n3_replay_a5.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "replay rc=$?"
python3 -I $E/summarize.py $OUT > $OUT/summary.md 2>&1; S "summary rc=$?"
S done
