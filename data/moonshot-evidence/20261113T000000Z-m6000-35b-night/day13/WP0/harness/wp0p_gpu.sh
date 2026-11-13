#!/usr/bin/env bash
# WP0 GPU sequence (one script, detached). Env from launcher (never stored): N3_GPU, N3_UNLOAD_URL. Optional N3_CUT_EPOCH, WAIT_UNTIL, FULL_UB.
# PROD-config check: singles prod s0/s1 (GEMM, F16 KV, cap1024, ub2048), then two-slot b3,prod;prod,b3.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; D=$M/occamy35-night-20261113/WP0/harness; export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$D/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; RES=$D/RESULT_PROD.md; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^n3-' | xargs -r docker rm -f >/dev/null 2>&1
  python3 -I $D/n3_analyze.py $OUT > $OUT/analyze.log 2>&1; python3 -I $D/wp0_report.py $OUT $RES > $OUT/report.log 2>&1 || echo "REPORT FAILED; see $OUT/steps.log ($(tail -1 $OUT/steps.log))" >> $RES
  S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock
LEFT=$(( ${WAIT_UNTIL:-0} - $(date +%s) )); [ $LEFT -gt 0 ] || LEFT=1
flock -w $LEFT 9 || { S "lock busy past deadline; skipped"; exit 90; }
S "lock taken"
[ -n "${N3_CUT_EPOCH:-}" ] && [ "$(date +%s)" -ge "$N3_CUT_EPOCH" ] && { S "past cut"; exit 5; }
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json; S "unload rc=$?"
for i in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU has compute apps; abort"; exit 4; }
N3_SINGLES=prod:0,prod:1 python3 -I $D/n3_replay.py $OUT --single --single-only > $OUT/single.log 2>&1; S "slots+single rc=$?"
N3_BLOCK_ORDERS="b3,prod;prod,b3" python3 -I $D/n3_replay.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "main rc=$?"
