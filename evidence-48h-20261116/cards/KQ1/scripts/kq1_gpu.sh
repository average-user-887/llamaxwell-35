#!/usr/bin/env bash
# KQ1 GPU chain (launch by a runner, not by the coder): (a) backend-ops FLASH_ATTN_EXT eval, (b) decode micro F16 vs Q8 K (kill if <1.05x at 96K),
# (c) KLD K vs F at 32K + 96K, (d) N3 short ABBA 6 pairs prod vs kq1, (e) counters. Output run-*/ ; summary lines in run-*/steps.log.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model unload POST only), N3_PORT, N3_HOST. Never GET unload-all. No GGML_CUDA_MAXWELL_GRAPHS.
# Optional: KQ1_PAIRS (default 6), KQ1_SKIP (space list of a b c d), KQ1_KLD_DOCS (default 0,1,2,3), KQ1_FP=1 (also run Fp noise arm).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}" "${N3_HOST:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM GGML_CUDA_FA_MAXWELL_GEMM_FUSED GGML_CUDA_OCCAMY_FA GGML_CUDA_OCCAMY_KVQ8 GGML_CUDA_OCCAMY_KVQ8_STAGE GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX LLAMA_N3_LEDGER_SYNC N3_EXTRA_ENV N3_NT_DIR
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/KQ1; PROD=$M/prod-libs/batch8-20261115; BIN=$E/build/bin
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
FREE=$(df -BG --output=avail $M | tail -1 | tr -dc 0-9); [ "$FREE" -ge 15 ] || { echo "disk ${FREE}G <15G; stop"; exit 3; }
export N3_GPU N3_PORT N3_HOST PYTHONDONTWRITEBYTECODE=1
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
SKIP=" ${KQ1_SKIP:-} "; skip() { [[ "$SKIP" == *" $1 "* ]]; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nkq-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
sha256sum $BIN/libggml-cuda.so $BIN/test-backend-ops $BIN/llama-server $PROD/libggml-cuda.so > $OUT/libs.sha256

# prod batch8 GGML env (from the prod launch script; asserted free of MAXWELL_GRAPHS / KVQ8)
PENV=$(grep -o -e '-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | sed 's/^-e //' | sort -u)
echo "$PENV" | grep -q MAXWELL_GRAPHS && { S "prod env has MAXWELL_GRAPHS; abort"; exit 5; }
echo "$PENV" | grep -q KVQ8 && { S "prod env has KVQ8; abort"; exit 5; }
TBO() {  # timeout_s, name, extra env..., -- test-backend-ops args
  local to=$1 n=$2; shift 2; local ev=(); while [ "$1" != -- ]; do ev+=(-e "$1"); shift; done; shift
  local pe=(); for x in $PENV; do pe+=(-e "$x"); done
  timeout $to docker run --rm --name nkq-tbo-$n --network none -u "$(id -u):$(id -g)" --device nvidia.com/gpu=$N3_GPU -v $BIN:/app:ro -e LD_LIBRARY_PATH=/app \
    "${pe[@]}" "${ev[@]}" --entrypoint /app/test-backend-ops $IMG "$@"
}
PF='hsk=256,hsv=256,nh=2,nr23=\[8,[12]\],kv=(24576|49152|98304),nb=1,'

if ! skip a; then  # (a) correctness: all FLASH_ATTN_EXT eval cases (incl. KQ1 q8-K cases) vs CPU, staged path on
  TBO 3600 eval-stage GGML_CUDA_OCCAMY_KVQ8=1 GGML_CUDA_OCCAMY_KVQ8_STAGE=1 -- -m test -o FLASH_ATTN_EXT > $OUT/a_eval_stage.log 2>&1; S "a eval stage rc=$?"
  TBO 3600 eval-plain GGML_CUDA_OCCAMY_KVQ8=1 -- -m test -o FLASH_ATTN_EXT > $OUT/a_eval_plain.log 2>&1; S "a eval plain-kvq8 rc=$?"
  for f in $OUT/a_eval_*.log; do S "a $(basename $f): $(grep -c ' OK' $f) OK, $(grep -c 'FAIL' $f) FAIL; $(grep -o 'occamy_kq1:.*' $f | tail -1)"; done
fi

if ! skip b; then  # (b) decode micro: F16 rows unaffected by KVQ8 flags; q8 rows = staged / plain kvq8 / stock
  TBO 1800 perf-stage GGML_CUDA_OCCAMY_KVQ8=1 GGML_CUDA_OCCAMY_KVQ8_STAGE=1 -- -m perf -o FLASH_ATTN_EXT -p "$PF" > $OUT/b_perf_stage.log 2>&1; S "b perf stage rc=$?"
  TBO 1800 perf-plain GGML_CUDA_OCCAMY_KVQ8=1 -- -m perf -o FLASH_ATTN_EXT -p "$PF" > $OUT/b_perf_plain.log 2>&1; S "b perf plain rc=$?"
  TBO 1800 perf-prod -- -m perf -o FLASH_ATTN_EXT -p "$PF" > $OUT/b_perf_prod.log 2>&1; S "b perf prod rc=$?"
  python3 -I - $OUT > $OUT/b_micro.md 2>&1 <<'PY'
import re, sys
out = sys.argv[1]; rx = re.compile(r'nr23=\[8,(\d)\],kv=(\d+),nb=1,.*?type_K=(\w+),type_V=(\w+).*?([\d.]+) us/run')
T = {}
for arm in ('stage', 'plain', 'prod'):
    for l in open('%s/b_perf_%s.log' % (out, arm), errors='replace'):
        m = rx.search(l)
        if m: T[(arm, int(m[1]), int(m[2]), m[3])] = float(m[5]) / 1000
print('| rows | kv | F16 ms (stage run) | Q8 staged ms | Q8 plain-kvq8 ms | Q8 stock ms | F16/Q8staged |\n|---|---|---|---|---|---|---|')
kill = None
for r in (1, 2):
    for kv in (24576, 49152, 98304):
        f = T.get(('stage', r, kv, 'f16')); q = T.get(('stage', r, kv, 'q8_0')); p = T.get(('plain', r, kv, 'q8_0')); s = T.get(('prod', r, kv, 'q8_0'))
        ratio = f / q if f and q else float('nan'); g = lambda x: '%.4f' % x if x else 'n/a'
        print('| %d | %d | %s | %s | %s | %s | %.3fx |' % (r, kv, g(f), g(q), g(p), g(s), ratio))
        if kv == 98304: kill = (kill or []) + [ratio]
ok = bool(kill) and all(x == x and x >= 1.05 for x in kill)
print('\nmicro gate (F16/Q8staged >= 1.05x at 96K, 1-row and 2-row): %s %s' % ('PASS' if ok else 'KILL', kill))
PY
  S "b micro: $(tail -1 $OUT/b_micro.md)"
  for f in $OUT/b_perf_*.log; do S "b $(basename $f): $(grep -o 'occamy_kq1:.*' $f | tail -1)"; done
  grep -q 'micro gate.*KILL' $OUT/b_micro.md && { S "KILL at micro gate; skipping c,d"; S done; exit 0; }
fi

if ! skip c; then  # (c) KLD vs F16 at 32K and 96K
  K=$OUT/kld; mkdir -p $K; export KQ1_GPU=$N3_GPU; D=${KQ1_KLD_DOCS:-0,1,2,3}
  for a in F K Kg $([ "${KQ1_FP:-0}" = 1 ] && echo Fp); do
    timeout 7200 python3 -I $E/kld/kq1_rows_harness.py $a $K $D > $K/arm_$a.log 2>&1; S "c kld arm $a rc=$?"
  done
  unset KQ1_GPU
  python3 -I $E/kld/kq1_kld_analyze.py $K > $K/analyze.log 2>&1; S "c kld analyze rc=$? $(grep -m1 Verdict $K/KQ1_KLD.md)"
fi

if ! skip d; then  # (d) N3 short, ABBA, KQ1_PAIRS pairs; each session = one driver call with one arm
  N=$OUT/n3; mkdir -p $N; export N3_PROFILE=short FULL_UB=2048 N3_CAP=1024 N3_EXTRA_ENV="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1"
  sess() {  # arm pair
    local a=$1 p=$2 d=$N/${1}_$2
    if [ $a = prod ]; then export N3_CAND_DIR=$PROD N3B_GEMM_LIB=$PROD/libggml-cuda.so; else export N3_CAND_DIR=$BIN N3B_GEMM_LIB=$BIN/libggml-cuda.so; fi
    N3_BLOCK_ORDERS=$a timeout 3300 python3 -I $E/n3_replay_kq1.py $d --no-sweep --no-ctxfull > $N/${a}_$p.log 2>&1; S "d $a pair $p rc=$?"
  }
  for p in $(seq 1 ${KQ1_PAIRS:-6}); do
    if [ $((p % 2)) = 1 ]; then sess prod $p; sess kq1 $p; else sess kq1 $p; sess prod $p; fi
  done
  for d in $N/*_[0-9]*/; do python3 -I $M/occamy35-night-20261113/WP0/harness/n3_analyze.py $d > $d/analyze.md 2>&1; done
  python3 -I $E/kq1_n3_pair.py $N > $OUT/n3_summary.md 2>&1; S "d n3 pair rc=$?"; grep -E '^- (engagement|N3 makespan|prefill)' $OUT/n3_summary.md | while read l; do S "d $l"; done
fi
# (e) counters: every kq1 server log must show occamy_kq1 stage>0 and occamy_fa_gemm kq8>0 (checked in kq1_n3_pair.py); raw lines here
grep -rho 'occamy_kq1:.*\|occamy_fa_gemm:.*' $OUT --include=server.log --include='log_*.txt' 2>/dev/null | sort | uniq -c > $OUT/e_counters.txt; S "e counters lines $(wc -l < $OUT/e_counters.txt)"
S done
