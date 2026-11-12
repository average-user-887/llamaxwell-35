#!/bin/bash
D=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/SMOKE3
exec 9>/media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock
flock -n 9 || { echo LOCK_BUSY; exit 9; }
for a in A B; do bash $D/raw/cyc.sh $a $a; done
docker rm -f smoke3-cyc >/dev/null 2>&1; echo end $(date)
