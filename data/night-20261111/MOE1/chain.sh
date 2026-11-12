#!/bin/bash
# MOE1 GPU chain, run under flock on gpu-bench.lock: G1 then G4 warm_decode + d4096
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/MOE1
echo "start $(date)"; nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw --format=csv,noheader -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
$R/g1.sh > $R/g1.log 2>&1; cat $R/g1.log
grep -q "bit_mismatch=0 .*BITEQUAL" $R/g1-B1.cmp.txt || { echo "G1 not BITEQUAL, stop"; exit 3; }
$R/g4.sh $R/cfg-moe1l3.json warm_decode
$R/g4.sh $R/cfg-moe1l3.json d4096
echo "end $(date)"
