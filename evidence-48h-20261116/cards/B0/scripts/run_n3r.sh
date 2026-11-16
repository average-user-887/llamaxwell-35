#!/bin/bash
# one command: run_n3r.sh URL ARM [n3r_replay flags...]   e.g. run_n3r.sh http://127.0.0.1:18583 batch8 --ctx 98304 --server-log /path/server.log
# Holds the GPU bench lock like the other harness scripts (set N3R_LOCK_HELD=1 if the caller already holds it). Out dir: $N3R_OUT or B0/runs/<arm>-<utc>.
M=<M6000_ROOT>; H=$(dirname "$(readlink -f "$0")")
[ $# -ge 2 ] || { echo "usage: $0 URL ARM [flags]"; exit 64; }
URL=$1; ARM=$2; shift 2
if [ -z "$N3R_LOCK_HELD" ]; then exec 9>$M/gpu-bench.lock; flock 9; fi
OUT=${N3R_OUT:-$H/runs/$ARM-$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$OUT"
python3 -I "$H/n3r_replay.py" --url "$URL" --arm "$ARM" --out "$OUT" "$@" 2>&1 | tee "$OUT/driver.log"; rc=${PIPESTATUS[0]}
echo "out=$OUT rc=$rc"; exit $rc
