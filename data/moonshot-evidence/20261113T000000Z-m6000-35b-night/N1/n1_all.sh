#!/usr/bin/env bash
M=<LAB>; N=$M/occamy35-night-20261113/N1
exec 9>$M/gpu-bench.lock; flock -w 5400 9 || { echo lock busy; exit 2; }
export N1_GPU=<gpu>
curl -s -m60 -X POST <lab-endpoint>/api/models/unload/qwen36-35b-moe; echo
sleep 8
for ARM in ${ARMS:-A B}; do bash $N/n1_run.sh $N/run $ARM; echo "arm $ARM rc=$?"; done
docker rm -f n1-srv >/dev/null 2>&1
