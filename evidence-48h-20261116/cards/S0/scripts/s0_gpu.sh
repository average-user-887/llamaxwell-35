#!/usr/bin/env bash
# S0 service soak: 60 min (SOAK_MIN) of prod batch8 (libs prod-libs/batch8-20261115, env/flags of run-35b-batch8-20261115.sh via n3_replay_l0 'prod' arm,
# cap1024 ub2048 b2048 2x~96K, PREFILL_SINGLE_STREAM) with long-context turns on both slots, mid-prefill cancels, slot save/restore + greedy-64 identity check.
# Own container ns0-*. Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (POST unload of qwen36-35b-moe only). Optional SOAK_MIN (default 60).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER_SYNC
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/S0; L=$M/occamy35-48h-20261115/L0; LIBS=$M/prod-libs/batch8-20261115
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/slots; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() {
  docker ps -a --format '{{.Names}}' | grep '^ns0-' | xargs -r docker rm -f >/dev/null 2>&1
  # slot files are written by the container as root: remove them (our own output, ~4 GB) via the server image
  docker run --rm --entrypoint /bin/rm -v $OUT/slots:/s sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d -f /s/s0_slot0.bin /s/s0_slot1.bin >/dev/null 2>&1
  S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
sha256sum $LIBS/llama-server $LIBS/libggml-cuda.so > $OUT/libs.sha256
export N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048 N3_CAP=1024 FULL_UB=2048 SOAK_OUT=$OUT SOAK_MIN=${SOAK_MIN:-60}
python3 -I $E/soak.py $OUT > $OUT/soak.log 2>&1; S "soak rc=$?"
python3 -I $E/summarize.py $OUT > $OUT/RESULT.md 2>&1; S "summary rc=$?"
cp $OUT/RESULT.md $E/RESULT_LATEST.md
S done
