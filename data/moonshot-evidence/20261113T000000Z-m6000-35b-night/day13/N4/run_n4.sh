#!/usr/bin/env bash
set -u
M=<LAB>
NC3=$M/occamy35-night-20261113/NC3
N4=$M/occamy35-night-20261113/N4
GPU=<gpu>
UNLOAD_URL=http://<lab-endpoint>/api/models/unload/qwen36-35b-moe
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
V4BIN=/m/codex-investigation-20261112/prefill/gemm/integration-v4/src/build/bin
G1BBIN=/m/occamy35-day-20261112/evidence/G1b/build/bin

unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM

cleanup() {
  docker ps -a --format '{{.Names}}' | grep -E '^n4-' | xargs -r docker rm -f >/dev/null 2>&1
}
trap cleanup EXIT

mkdir -p "$NC3/dump" "$NC3/out_stock" "$NC3/out_gemm" "$N4"

exec 9>"$M/gpu-bench.lock"
flock -n 9 || { echo "gpu-bench.lock busy"; exit 2; }

curl -s -m60 -X POST "$UNLOAD_URL" > "$N4/unload.json"
n=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -eq 0 ] || { echo "GPU has $n compute apps; abort"; exit 4; }

echo "=== N4 step 1: n4_dump $(date -Is) ===" > "$N4/run.log"
docker run --rm --name n4-dump-$$ \
  --device "nvidia.com/gpu=$GPU" \
  -v <MODELS>:/models:ro \
  -v "$M:/m" \
  -e "LD_LIBRARY_PATH=$V4BIN:$G1BBIN" \
  -e GGML_CUDA_OCCAMY_FA=1 \
  -e GGML_CUDA_OCCAMY_FUSED=1 \
  -e GGML_CUDA_OCCAMY_FA_PIPE=3 \
  -e GGML_CUDA_OCCAMY_FA_ADDR=1 \
  --entrypoint /m/occamy35-night-20261113/NC3/build/n4_dump \
  "$IMG" \
  /models/occamy-1.0-Q4_0-headq40b.gguf \
  /m/occamy35-night-20261113/NC3/dump \
  98500 \
  /m/occamy35-night-20261113/NC3/d0.i32 \
  3,19,39 \
  4096:1024,4160:64,32768:1024,33024:256,98048:1024,98304:256 \
  >> "$N4/run.log" 2>&1
rc_dump=$?
echo "n4_dump rc=$rc_dump $(date -Is)" >> "$N4/run.log"
[ "$rc_dump" -eq 0 ] || exit "$rc_dump"

echo "=== N4 step 2: n4_drive $(date -Is) ===" >> "$N4/run.log"
docker run --rm --name n4-drive-$$ \
  --device "nvidia.com/gpu=$GPU" \
  -v "$M:/m" \
  -e "LD_LIBRARY_PATH=$V4BIN" \
  --entrypoint bash \
  "$IMG" -c '
set -e
NC3=/m/occamy35-night-20261113/NC3
for S in 4096 4160 32768 33024 98048 98304; do
  case "$S" in
    4096|32768|98048) ROWS="0,1,511,1022,1023" ;;
    4160) ROWS="0,1,32,62,63" ;;
    33024|98304) ROWS="0,1,128,254,255" ;;
  esac
  for L in 3 19 39; do
    mkdir -p "$NC3/out_stock/stop$S" "$NC3/out_gemm/stop$S"
    "$NC3/build/n4_drive" "$NC3/dump/stop$S" "$L" "$NC3/out_stock/stop$S"
    GGML_CUDA_FA_MAXWELL_GEMM=1 "$NC3/build/n4_drive" "$NC3/dump/stop$S" "$L" "$NC3/out_gemm/stop$S" stages "$ROWS"
  done
done
' >> "$N4/run.log" 2>&1
rc_drive=$?
echo "n4_drive rc=$rc_drive $(date -Is)" >> "$N4/run.log"

# Release lock before CPU FP64 comparison
flock -u 9
exec 9>&-

echo "=== N4 step 3: ref_fa64 compare $(date -Is) ===" >> "$N4/run.log"
python3 -I "$NC3/n4_analyze.py" >> "$N4/run.log" 2>&1
rc_an=$?
echo "n4_analyze rc=$rc_an $(date -Is)" >> "$N4/run.log"
exit "$rc_an"
