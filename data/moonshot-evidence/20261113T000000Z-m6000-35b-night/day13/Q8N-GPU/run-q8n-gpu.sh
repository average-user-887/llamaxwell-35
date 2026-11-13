#!/usr/bin/env bash
# Q8N-GPU card: whole GPU sequence in one script. Env (passed by the launcher, never stored): Q8N_GPU, Q8N_UNLOAD_URL. Optional SMOKE=1, DOCS (0,1,2,3), CAP_MIN (75).
# Takes gpu-bench.lock itself (waits), POSTs unload (qwen36-35b-moe only), runs exact -> restore -> ABBA -> nvprof -> single/append, writes GPU_RESULT.md in trap EXIT, removes nq8g-* containers, releases lock.
set -u
: "${Q8N_GPU:?}" "${Q8N_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$Q8N_UNLOAD_URL" in *unload-all*) echo "refusing unload-all"; exit 2;; *qwen36-35b-moe*) ;; *) echo "bad unload url"; exit 2;; esac
M=<LAB>; Q=$M/occamy35-night-20261113/Q8N; G=$Q/gates; export Q8N_GPU PYTHONDONTWRITEBYTECODE=1
OUT=$G/gpurun-$(date -u +%Y%m%dT%H%M%SZ)${SMOKE:+-smoke}; mkdir -p $OUT; RES=$Q/GPU_RESULT.md; [ "${SMOKE:-0}" = 1 ] && RES=$OUT/GPU_RESULT_smoke.md
step() { echo "$* $(date -u +%T)" >> $OUT/steps.log; }
cleanup() { kill ${TP:-} 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^nq8g-' | xargs -r docker rm -f >/dev/null 2>&1
  python3 -I $G/q8n_report.py $OUT $RES >> $OUT/report.log 2>&1; step "cleanup, report -> $RES"; exec 9>&-; step "lock released"; }
trap cleanup EXIT
[ "$(sha256sum $Q/build/bin/libggml-cuda.so | cut -c1-8)" = d2886ab9 ] || { echo "lib sha mismatch"; step "lib sha mismatch"; exit 3; }
exec 9>$M/gpu-bench.lock; flock -w 3600 9 || { step "lock timeout"; exit 90; }
T0=$(date +%s); CAP=${CAP_MIN:-75}; step "start lock taken lib=$(sha256sum $Q/build/bin/libggml-cuda.so | cut -c1-16) smoke=${SMOKE:-0}"
curl -s -m60 -X POST "$Q8N_UNLOAD_URL" > $OUT/unload.out 2>&1; step "unload rc=$?"
apps() { nvidia-smi -i "$Q8N_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l; }
waitidle() { for i in $(seq 1 120); do [ "$(apps)" = 0 ] && return 0; sleep 1; done; return 1; }
waitidle || { step "compute apps present; abort"; exit 91; }
nvidia-smi -i "$Q8N_GPU" --query-gpu=clocks.sm,clocks.mem,power.draw,memory.used --format=csv -l 1 > $OUT/telemetry.csv & TP=$!
left() { echo $(( T0 + CAP*60 - $(date +%s) )); }
DOCS=${DOCS:-0,1,2,3}
# 1 exactness: doc-major so a cut still leaves paired data
if [ "${SMOKE:-0}" = 1 ]; then export Q8N_BUCKETS=2048,4096 Q8N_NR=8; DOCS=0; else export Q8N_BUCKETS=32768,98304; fi  # doc 0 both buckets, later docs 32K only (98K prefill is ~8 min per arm-doc)
export Q8N_CUT_EPOCH=$(( T0 + 42*60 ))
for d in ${DOCS//,/ }; do [ "${SMOKE:-0}" != 1 ] && { if [ "$d" = 0 ]; then export Q8N_BUCKETS=32768,98304; else export Q8N_BUCKETS=32768; fi; }; [ $(date +%s) -gt $Q8N_CUT_EPOCH ] && { step "exact cut before doc $d"; break; }
  for a in G N Gk Nk; do waitidle && { PYTHONDONTWRITEBYTECODE=1 python3 -B $G/q8n_rows_harness.py $a $OUT/exact $d >> $OUT/run.log 2>&1; step "rows $a doc=$d rc=$?"; }; done; done
Q8N_BUCKETS=$([ "${SMOKE:-0}" = 1 ] && echo 2048,4096 || echo 32768,98304) python3 -I $G/q8n_analyze.py $OUT/exact > $OUT/exact_analyze.out 2>&1; step "exact analyze rc=$?"; unset Q8N_BUCKETS Q8N_NR Q8N_CUT_EPOCH
# 2 slot restore: q8 slot files + the restore test were produced by the SMOKE run of this same lib (restore.json copied); no re-build here
if [ "${SMOKE:-0}" = 1 ]; then waitidle && { python3 -I $G/q8n_restore.py $OUT/restore >> $OUT/run.log 2>&1; step "restore rc=$?"; }
else mkdir -p $OUT/restore; cp $G/restore_from_smoke.json $OUT/restore/restore.json; step "restore.json copied from smoke run"; fi
# 4 nvprof
{ waitidle && { python3 -I $G/q8n_prof.py $OUT/prof F,Q,K >> $OUT/run.log 2>&1; step "prof rc=$?"; }; } || step "prof skipped (cap)"
# 3 two-slot deep ABBA x3: F,Q(K+V),K(K-only) counterbalanced
export Q8N_CUT_EPOCH=$(( T0 + 62*60 )); ORD=FQKKQF,KQFFQK,FQKKQF; SM=""
[ "${SMOKE:-0}" = 1 ] && { ORD=FQK; SM=--smoke; }
waitidle && { Q8N_UNITS=both_deep python3 -I $G/q8n_srv.py $OUT/srv "$ORD" $SM >> $OUT/run.log 2>&1; step "srv rc=$?"; }
python3 -I $G/q8n_srv_analyze.py $OUT/srv > $OUT/srv_analyze.out 2>&1; step "srv analyze rc=$?"; unset Q8N_CUT_EPOCH
# 5 single 98K / append only inside the cap
if [ "$(left)" -gt 600 ] && [ "${SMOKE:-0}" != 1 ]; then export Q8N_CUT_EPOCH=$(( T0 + (CAP-2)*60 ))
  waitidle && { Q8N_UNITS=single_98k,append_4k python3 -I $G/q8n_srv.py $OUT/srv5 "FQQF,QFFQ" >> $OUT/run.log 2>&1; step "srv5 rc=$?"; }
  python3 -I $G/q8n_srv_analyze.py $OUT/srv5 > $OUT/srv5_analyze.out 2>&1
else step "single/append skipped (cap)"; fi
step "end elapsed_min=$(( ($(date +%s) - T0)/60 ))"
