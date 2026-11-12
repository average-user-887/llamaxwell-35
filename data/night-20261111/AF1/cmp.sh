#!/bin/bash
# cmp.sh <A> <B> : compare dumps in raw/dumps; append to raw/<B>.cmp.txt
W=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110; D=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/dumps; O=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/$2.cmp.txt
{ cd /tmp; python3 -I $W/tests/compare_report.py logits $D/$1.bin $D/$2.bin --out /media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/AF1/$2.logits.json
for s in step64 step2112; do echo "state $s: A=$(sha256sum $D/$1.$s.state|cut -c1-12) B=$(sha256sum $D/$2.$s.state|cut -c1-12) diffbytes=$(cmp -l $D/$1.$s.state $D/$2.$s.state | wc -l)"; done; } 2>&1 | tee $O
