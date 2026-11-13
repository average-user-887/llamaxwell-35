#!/usr/bin/env bash
# N3 overnight GPU run. Holds gpu-bench.lock throughout. Env required: N3_GPU (device id/uuid), N3_UNLOAD_URL (full POST .../api/models/unload/qwen36-35b-moe).
# Optional: N3_CUT_EPOCH (stop launching sessions within ~11 min of it), N3_PROFILE=full|short, N3_START/N3_TURNS/N3_OFFSET_S, N3_BLOCK_ORDERS, N3_SWEEP_ORDER, N3_BIN_<ARM>.
# Flags pass to n3_replay.py: --smoke (slots + 1 b3 session, 2 turns), --slots-only, --no-sweep, --no-ctxfull.   --plan prints schedule only (no lock, no GPU).
# Arms: gdn4 = image /app binary + gdn4 lib + run-35b-gdn4 env; b3 / cap1024 / gemm = G1b build (cap patch) + bundle3 lib + bundle3 env (cap1024: --mixed-batch-size 1024; gemm: +GGML_CUDA_FA_MAXWELL_GEMM=1).
# Own containers: n3-* only; removed at exit by trap. Never touches llamacpp-m6000*.
set -u
M=<LAB>; D=$M/occamy35-night-20261113/N3
[ "${1:-}" = "--plan" ] && exec python3 -I $D/n3_replay.py --plan
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
cleanup() { docker ps -a --format '{{.Names}}' | grep '^n3-' | xargs -r docker rm -f >/dev/null 2>&1; pkill -f "query-gpu=clocks.sm,clocks.mem,power.draw" 2>/dev/null; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -n 9 || { echo "gpu-bench.lock busy"; exit 2; }
[ -x $M/occamy35-day-20261112/evidence/G1b/build/bin/llama-server ] || { echo "no G1b build"; exit 3; }
if [ -n "${N3_CUT_EPOCH:-}" ] && [ "$(date +%s)" -ge "$N3_CUT_EPOCH" ]; then echo "past N3_CUT_EPOCH; not starting"; exit 5; fi
OUT=$D/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT $D/slots
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
n=$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "GPU has $n compute apps; abort"; exit 4; }
export N3_GPU
python3 -I $D/n3_replay.py $OUT "$@" > $OUT/run.log 2>&1; rc=$?
echo "rc=$rc out=$OUT"
python3 -I $D/n3_analyze.py $OUT > $OUT/analyze.log 2>&1
exit $rc
