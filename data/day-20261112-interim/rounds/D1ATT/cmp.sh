#!/bin/bash
# cmp.sh <A> <B> [states...]
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D1ATT; X=$R/dumps; a=$1 b=$2; shift 2
{ cd /tmp; python3 -I $W/tests/compare_report.py logits $X/$a.bin $X/$b.bin --out $R/$b.logits.json
for s in "$@"; do echo "state $s: A=$(sha256sum $X/$a$s|cut -c1-12) B=$(sha256sum $X/$b$s|cut -c1-12) diffbytes=$(cmp -l $X/$a$s $X/$b$s | wc -l)"; done; } 2>&1 | tee $R/$b.cmp.txt
