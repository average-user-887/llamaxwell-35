#!/bin/bash
# locked.sh "<steps>" : take gpu-bench.lock, unload 35B from llama-swap, run c5.sh steps, release.
LOCK=<STORAGE>/m6000-llama-cpp/gpu-bench.lock; C5=<HOME>/.claude/jobs/fc533454/tmp/c5
exec 9>>$LOCK; flock -w 1200 9 || { echo "lock timeout"; exit 75; }
echo "lock taken $(date +%T); running: $(curl -s -m10 ${SWAP_URL:?see LAB_FACTS}/running)"
for m in qwen36-35b-moe qwen38-27b-dense; do curl -s -m60 -X POST ${SWAP_URL:?see LAB_FACTS}/api/models/unload/$m -o /dev/null -w "unload $m %{http_code}\n"; done
for i in $(seq 1 60); do docker ps --format '{{.Names}}' | grep -qx 'llamacpp-m6000\|llamacpp-m6000-27b' || break; sleep 2; done
docker ps --format '{{.Names}}' | grep -x 'llamacpp-m6000\|llamacpp-m6000-27b' && { echo "prod still up, abort"; exit 3; }
nvidia-smi -i GPU-REDACTED --query-gpu=memory.used,power.limit --format=csv,noheader
shift 0; for s in "$@"; do eval "$s"; done
echo "released $(date +%T)"
