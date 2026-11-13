#!/usr/bin/env bash
# N3B whole-GPU sequence, one script. Env from launcher (never stored): N3_GPU, N3_UNLOAD_URL. Optional N3_CUT_EPOCH, WAIT_UNTIL (epoch; give up on lock after this).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; D=$M/occamy35-night-20261113/N3B; export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$D/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; RES=$D/RESULT.md; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { kill ${TP:-} 2>/dev/null; docker ps -a --format '{{.Names}}' | grep '^n3-' | xargs -r docker rm -f >/dev/null 2>&1
  if [ -f $OUT/runs/b0_p0_b3/run.json ] || [ -f $OUT/runs/prof_gemm/run.json ]; then python3 -I $D/n3_analyze.py $OUT > $OUT/analyze.log 2>&1; python3 -I $D/n3b_report.py $OUT $RES >> $OUT/report.log 2>&1
  else echo "FAILED/SKIPPED: see $OUT/steps.log ($(tail -1 $OUT/steps.log))" > $RES; fi
  S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock
LEFT=$(( ${WAIT_UNTIL:-0} - $(date +%s) )); [ $LEFT -gt 0 ] || LEFT=1
flock -w $LEFT 9 || { S "lock busy past deadline; skipped"; exit 90; }
S "lock taken"
[ "$(date +%s)" -ge "${N3_CUT_EPOCH:-0}" ] && [ -n "${N3_CUT_EPOCH:-}" ] && { S "past cut"; exit 5; }
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json; S "unload rc=$?"
for i in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU has compute apps; abort"; exit 4; }
# 1 nvprof gemm engagement session (also acts as gemm smoke): 2 turns
python3 -I $D/n3_replay.py $OUT --prof --prof-only > $OUT/prof.log 2>&1; S "prof rc=$?"
# 2 main: 3 blocks, b3+gemm
N3_BLOCK_ORDERS="b3,gemm;gemm,b3;b3,gemm" python3 -I $D/n3_replay.py $OUT --no-sweep --no-ctxfull > $OUT/run.log 2>&1; S "main rc=$?"
