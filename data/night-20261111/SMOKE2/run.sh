#!/bin/bash
D=<STORAGE>/m6000-llama-cpp/occamy35-night-20261111/rounds/SMOKE2
exec 9><STORAGE>/m6000-llama-cpp/gpu-bench.lock
flock -n 9 || { echo LOCK_BUSY; exit 9; }
echo start $(date); journalctl -k --since "now" >/dev/null 2>&1
nvidia-smi -i GPU-REDACTED --query-compute-apps=pid --format=csv,noheader | wc -l
bash $D/raw/arm.sh A; echo A rc $?
bash $D/raw/arm.sh B; echo B rc $?
docker rm -f smoke2-serve-A smoke2-serve-B >/dev/null 2>&1
echo end $(date)
