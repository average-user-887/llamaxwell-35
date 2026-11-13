#!/bin/bash
# G1c runner. Env (never stored in files): G1C_GPU (device id), G1C_UNLOAD_URL (full POST URL for unloading qwen36-35b-moe only). Optional: TIER2=0 to skip Ap/C, SKIP_FGDIV=1, CUT_LOCAL='22:30' (CET).
# Rules: flock gpu-bench.lock; POST unload of qwen36-35b-moe only (never GET unload-all); abort if compute apps present; containers g1c-*;
# GGML_CUDA_MAXWELL_GRAPHS absent+unset; no power/clock changes.
set -u
: "${G1C_GPU:?set G1C_GPU}" "${G1C_UNLOAD_URL:?set G1C_UNLOAD_URL}" ; [ "${SKIP_FGDIV:-0}" = 1 ] || : "${G1C_PUBLISH:?set G1C_PUBLISH (docker -p bind:port)}" "${G1C_PORT:?set G1C_PORT (same port)}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$G1C_UNLOAD_URL" in *unload-all*) echo "refusing unload-all"; exit 2;; *qwen36-35b-moe*) ;; *) echo "unload URL must name qwen36-35b-moe"; exit 2;; esac
M=<LAB>; G=$M/occamy35-day-20261112/evidence/G1c
OUT=$G/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; export G1C_GPU
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo "gpu-bench.lock busy"; exit 90; }
echo "start $(date -u +%FT%TZ)" > $OUT/steps.log
curl -s -m60 -X POST "$G1C_UNLOAD_URL" > $OUT/unload.out 2>&1; echo "unload rc=$?" >> $OUT/steps.log
apps() { nvidia-smi -i "$G1C_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l; }
[ "$(apps)" = 0 ] || { echo "compute apps present; abort" >> $OUT/steps.log; exit 91; }
export G1C_CUT_EPOCH=${G1C_CUT_EPOCH:-$(TZ=Europe/Warsaw date -d "today ${CUT_LOCAL:-22:30}" +%s)}   # env override of the epoch
REM=$(( (G1C_CUT_EPOCH - $(date +%s)) / 60 ))
# MODE full: 8 docs x 64 rows (needs ~125 min incl. FGDIV); MODE cut: 4 docs x 128 rows (>=512 rows/bucket, ~75 min). Auto by minutes left before the cut.
if [ -n "${MODE:-}" ]; then :; elif [ "$REM" -ge 130 ]; then MODE=full; else MODE=cut; fi
if [ "$MODE" = full ]; then export G1C_NR=64; T1=0,1,2,3,4,5,6,7; export G1C_PER=2; else export G1C_NR=128; T1=0,1,2,3; export G1C_PER=1; fi
step() { echo "$* $(date -u +%T)" >> $OUT/steps.log; }
export G1C_DOCS=$T1
step "mode=$MODE rem_min=$REM nr=$G1C_NR docs=$T1"
waitidle() { for i in $(seq 1 120); do [ "$(apps)" = 0 ] && return 0; sleep 1; done; return 1; }   # bounded wait for previous container's context to vanish
run() { arm=$1; docs=$2; waitidle || { step "apps present before $arm; abort"; return 1; }
  PYTHONDONTWRITEBYTECODE=1 python3 -B $G/g1c_harness.py $arm $OUT $docs >> $OUT/run.log 2>&1; step "arm $arm docs=$docs rc=$?"; }
# Tier 1: gate arms, 8 docs = 512 rows per bucket (A first: it is the reference for everything)
run A $T1
run B $T1
# FGDIV (<=20 min)
if [ "${SKIP_FGDIV:-0}" != 1 ] && [ "$(date +%s)" -lt "$G1C_CUT_EPOCH" ]; then
  waitidle && { PYTHONDONTWRITEBYTECODE=1 python3 -B $G/g1c_fgdiv.py $OUT/fgdiv AB >> $OUT/run.log 2>&1; step "fgdiv rc=$?"; }
fi
# Tier 2: noise floors on doc 0 only (64-128 rows/bucket), only while before the cut
if [ "${TIER2:-1}" = 1 ]; then
  [ "$(date +%s)" -lt "$G1C_CUT_EPOCH" ] && run Ap 0
  [ "$(date +%s)" -lt "$G1C_CUT_EPOCH" ] && run C 0
fi
docker ps -a --format '{{.Names}}' | grep '^g1c-' | xargs -r docker rm -f >/dev/null   # only our own containers
python3 -I $G/g1c_analyze.py $OUT > $OUT/analyze.out 2>&1; step "analyze rc=$?"
[ -d $OUT/fgdiv ] && python3 -I $G/fgdiv_analyze.py $OUT/fgdiv > $OUT/fgdiv_analyze.out 2>&1
step "end"
