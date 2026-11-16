#!/usr/bin/env bash
# F4: single-slot row (restore 84K slot, append to 90K then 94K, timed) on prod batch8 libs. n3_replay.py --single --single-only, N3_SINGLES=prod:0 x3 (repeat for noise).
# Copy of n3_replay.py with container prefix n28h- -> nf4- only. No MTP arm. Env (launch only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/F4
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nf4-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
S "waiting for lock"
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
LIBS=$M/prod-libs/batch8-20261115
export N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 FULL_UB=2048 N3_CAP=1024
for r in 1 2 3; do
  n=single_$r; mkdir -p $OUT/$n
  N3_SINGLES=prod:0 python3 -I $E/n3_replay_nf4.py $OUT/$n --single --single-only > $OUT/$n/run.log 2>&1; S "$n rc=$?"
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  S "$n engaged: $(grep -h 'n2gdn variant\|occamy FA gemm\|occamy_fa_gemm\|MOE_V3 launches\|n2gdn launches' $L | tail -4 | tr '\n' '|')"
done
S done
