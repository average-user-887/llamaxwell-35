#!/bin/bash
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/E07
echo "lock $(date)"; nvidia-smi --id=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-compute-apps=pid,used_memory --format=csv,noheader
bash $R/g4.sh $R/cfg-e07p3sanity.json d16384
for m in d16384 warm_decode d4096; do bash $R/g4.sh $R/cfg-e07p3.json $m; done
echo "CHAIN DONE $(date)"
