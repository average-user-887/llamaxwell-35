#!/usr/bin/env bash
# D0 determinism check: N3 short-profile replay x4 on PROD batch8 libs (prod-libs/batch8-20261115), greedy (temperature 0), per-slot generated token ids saved per turn
# (run.json reqs[].gen_ids). Then compare.py reports per-slot token identity between runs. Same env as L0 (PREFILL_SINGLE_STREAM, OCCAMY_DEBUG, cap1024, ub2048, b2048, 2x98K).
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only).   ~15 min wall.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_OFA_RAGGED GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER LLAMA_N3_SCHED
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/D0; LIBS=$M/prod-libs/batch8-20261115
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nd0-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
export N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
# one invocation = 4 sequential fresh-container sessions (blocks b0..b3, arm prod each); slots are restored from the read-only N3 slot files per session
FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod;prod;prod;prod" python3 -I $E/n3_replay_d0.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "replay rc=$?"
for d in $OUT/runs/*/; do grep -h "occamy_fa_gemm" $d/server.log 2>/dev/null | tail -1; done > $OUT/gemm_lines.txt
python3 -I $E/compare.py $(ls -d $OUT/runs/b*_prod) > $OUT/compare.md 2>&1; S "compare rc=$?"
S done
