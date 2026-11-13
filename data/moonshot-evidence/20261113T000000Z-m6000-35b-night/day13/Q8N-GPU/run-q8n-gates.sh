#!/usr/bin/env bash
# Q8N GPU gates. READY, NOT RUN by the Q8N coder. Env (never stored in files): Q8N_GPU (M6000 device id), Q8N_UNLOAD_URL (full POST URL
# <lab-endpoint>/api/models/unload/qwen36-35b-moe). Optional: PARTS="exact srv" (default both), DOCS (default 0,1,2,3,4,5,6,7), ORDER (default FQQF,QFFQ,FQQF),
# EXTRA_ARMS (exactness tier 2, default "Gp F X"; add "Nk Nv" for mixed instances), SMOKE=1 (1 doc / 1 server unit per arm).
# Rules: flock gpu-bench.lock; POST unload of qwen36-35b-moe only (never GET, never unload-all); abort if compute apps; no power/clock/paranoid changes;
# GGML_CUDA_MAXWELL_GRAPHS absent, GGML_CUDA_FA_MAXWELL_GEMM unset; only nq8g-* containers, removed at the end. Prod untouched.
set -u
: "${Q8N_GPU:?set Q8N_GPU}" "${Q8N_UNLOAD_URL:?set Q8N_UNLOAD_URL}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$Q8N_UNLOAD_URL" in *unload-all*) echo "refusing unload-all"; exit 2;; *qwen36-35b-moe*) ;; *) echo "unload URL must name qwen36-35b-moe"; exit 2;; esac
M=<LAB>; Q=$M/occamy35-night-20261113/Q8N; G=$Q/gates
[ -x $Q/build/bin/llama-server ] && [ -x $G/q8n_rows ] || { echo "missing build or q8n_rows (run build-q8n-rows.sh)"; exit 3; }
OUT=$G/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; export Q8N_GPU
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo "gpu-bench.lock busy"; exit 90; }
step() { echo "$* $(date -u +%T)" >> $OUT/steps.log; }
cleanup() { kill ${TP:-} 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^nq8g-' | xargs -r docker rm -f >/dev/null; step "cleanup"; }
trap cleanup EXIT
step "start lib_sha256=$(sha256sum $Q/build/bin/libggml-cuda.so | cut -c1-64)"
curl -s -m60 -X POST "$Q8N_UNLOAD_URL" > $OUT/unload.out 2>&1; step "unload rc=$?"
apps() { nvidia-smi -i "$Q8N_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l; }
[ "$(apps)" = 0 ] || { step "compute apps present; abort"; exit 91; }
nvidia-smi -i "$Q8N_GPU" --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv -l 1 > $OUT/telemetry.csv & TP=$!
waitidle() { for i in $(seq 1 120); do [ "$(apps)" = 0 ] && return 0; sleep 1; done; return 1; }
DOCS=${DOCS:-0,1,2,3,4,5,6,7}; ORDER=${ORDER:-FQQF,QFFQ,FQQF}; SM=""
[ "${SMOKE:-0}" = 1 ] && { DOCS=0; export Q8N_NR=8; ORDER=FQ; SM=--smoke; }
rows() { waitidle || { step "apps present before $1; abort"; return 1; }
  PYTHONDONTWRITEBYTECODE=1 python3 -B $G/q8n_rows_harness.py $1 $OUT/exact $2 >> $OUT/run.log 2>&1; step "rows $1 docs=$2 rc=$?"; }
if [[ " ${PARTS:-exact srv} " == *" exact "* ]]; then
  rows G $DOCS; rows N $DOCS                                   # gate arms, all docs
  for a in ${EXTRA_ARMS:-Gp F X}; do rows $a ${DOCS%%,*}; done  # noise / context / same-cache compare, first doc
  python3 -I $G/q8n_analyze.py $OUT/exact > $OUT/exact_analyze.out 2>&1; step "exact analyze rc=$?"
fi
if [[ " ${PARTS:-exact srv} " == *" srv "* ]]; then
  waitidle && { python3 -I $G/q8n_srv.py $OUT/srv "$ORDER" $SM >> $OUT/run.log 2>&1; step "srv rc=$?"; }
  python3 -I $G/q8n_srv_analyze.py $OUT/srv > $OUT/srv_analyze.out 2>&1; step "srv analyze rc=$?"
fi
step "end out=$OUT"; echo "out=$OUT"
