#!/bin/bash
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/HG1
echo "start $(date)"; $R/g1.sh > $R/g1.log 2>&1; cat $R/g1.log
grep -q "bit_mismatch=0 .*BITEQUAL" $R/g1-B1.cmp.txt || { echo "G1 not BITEQUAL, stop"; exit 3; }
$R/prof.sh; echo "end $(date)"
