#!/bin/bash
M=<LAB>; D=$M/occamy35-day-20261112
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo LOCK BUSY; exit 3; }
nvidia-smi -i <gpu> --query-compute-apps=pid,name --format=csv,noheader | grep -q . && { echo GPU BUSY; exit 4; }
curl -s -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
nvidia-smi -i <gpu> --query-compute-apps=pid,name --format=csv,noheader | grep -q . && { echo GPU BUSY after unload; exit 4; }
for a in B A; do python3 -I $D/evidence/E4/e4_run.py $a > $D/evidence/E4/run_$a.$(date -u +%H%M%S).log 2>&1; echo "arm $a rc=$?"; done
docker rm -f $(docker ps -aq --filter name=ev4-) 2>/dev/null
echo finished
