#!/bin/bash
# GDN4 G4 ABBA x5 (warm_decode, d4096). Holds gpu-bench.lock; per-model POST unload only.
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/GDN4; SWAP_URL=http://172.21.0.1:11445
exec 9>/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock; flock 9; echo "lock $(date +%T)"
curl -s -m10 $SWAP_URL/running | grep -q qwen36-35b-moe && { curl -s -m60 -X POST $SWAP_URL/api/models/unload/qwen36-35b-moe; echo; }
echo "running: $(curl -s -m10 $SWAP_URL/running)"
$R/g4.sh $R/cfg-gdn4.json warm_decode
$R/g4.sh $R/cfg-gdn4.json d4096
for a in A B; do echo "peak mem.used d4096 $a: $(cat $R/g4-gdn4/gdn4-d4096-*-$a-*.smi.csv | awk -F', ' 'NR>0{gsub(/ MiB/,"",$9); if($9+0>m)m=$9+0}END{print m}') MiB"; done
echo "ABBA-finished $(date +%T)"
