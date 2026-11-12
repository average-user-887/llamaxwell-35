#!/bin/bash
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/BUNDLE2
exec 9>/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock; flock 9; echo "lock $(date)"
$R/g1.sh > $R/g1.log 2>&1; cat $R/g1.log
grep -q "verdict=BITEQUAL" $R/g1-B1.cmp.txt && grep -q "A=652476ce477f B=652476ce477f" $R/g1-B1.cmp.txt && grep -q "A=5be5fc2d816d B=5be5fc2d816d" $R/g1-B1.cmp.txt || { echo G1-MISMATCH-STOP; exit 4; }
$R/abba.sh
echo CHAIN-DONE $(date)
