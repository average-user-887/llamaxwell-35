#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
start=$(date +%s)
card=$(cd -- "$(dirname -- "$0")" && pwd -P)
campaign=$(dirname "$card")
lab=$(dirname "$campaign")
runner=$campaign/runner
runtime=$campaign/sparse/r2/libs
source "$runner/night_trial_runtime.sh"
cd "$card"

case ${1:-} in
  --check|--run) [[ $# == 1 ]] || exit 2 ;;
  *) echo 'usage: release.sh --check | --run' >&2; exit 2 ;;
esac

sha256sum --check --quiet FROZEN.sha256
sha256sum --check --quiet INPUTS.sha256
python3 -B verify_build.py | cmp - build.json
python3 -B offline_check.py
python3 -B verify_probe.py
python3 -B verify_stride.py
night_profile
flags=("${night_cuda[@]}" -e LD_LIBRARY_PATH=/candidate
  -e OCCAMY_Q2_R2_FIXED_INPUT=1 -e GGML_CUDA_OCCAMY_FA_MROW=3
  -e GGML_CUDA_OCCAMY_FA2_NU=4 -e GGML_CUDA_OCCAMY_FA_SPARSE=1
  -e GGML_CUDA_OCCAMY_FA_SPARSE_R2=1 -e GGML_CUDA_OCCAMY_FA_SPARSE_R2_CHECK=0
  -e GGML_CUDA_OCCAMY_FA_SPARSE_PROFILE=0 -e GGML_CUDA_MV1=0)
printf '%s\n' "${flags[@]}" | python3 -c '
import sys
lines = sys.stdin.read().splitlines()
keys = [entry.split("=", 1)[0] for entry in lines[1::2]]
assert len(lines) == 2*len(keys) and all(flag == "-e" for flag in lines[::2])
assert len(set(keys)) == len(keys) and not any("TOP16" in key for key in keys)
'
if [[ $1 == --check ]]; then
  echo 'Frozen Q2 sparse stride CPU checks passed.'
  exit 0
fi

night_lock_gpu
deadline=$((start + 180))
[[ ${ROOT_GPU_DEADLINE_EPOCH:-} =~ ^[0-9]+$ ]] || { echo 'Missing root GPU lease deadline' >&2; exit 2; }
(( deadline <= ROOT_GPU_DEADLINE_EPOCH )) || { echo 'operator stage would extend past the GPU lease' >&2; exit 124; }
id=$(date -u +%Y%m%dT%H%M%SZ)-$$
out=$card/receipts/$id
mkdir -p "$out"
printf 'Q2_R2_STRIDE_RECEIPT=%s\n' "$out"
current=

clean() {
  local ids
  [[ -n $current ]] || return 0
  ids=$(timeout -k 1s 3s docker ps -aq --filter "name=^/$current$" --filter "label=q2r2stride.owner=$id") || return 1
  if [[ -n $ids ]]; then
    [[ $ids =~ ^[0-9a-f]+$ ]] || return 1
    timeout -k 1s 5s docker rm -f "$ids" >> "$out/cleanup.log" 2>&1 || return 1
  fi
  ids=$(timeout -k 1s 3s docker ps -aq --filter "name=^/$current$") || return 1
  [[ -z $ids ]] || return 1
  current=
}

finish() {
  local rc=$? cleanup_rc=0 done
  trap - EXIT TERM INT
  clean || cleanup_rc=$?
  done=$(date +%s)
  (( done <= deadline )) || rc=124
  printf 'runner_exit=%d\ncleanup_exit=%d\nstarted_epoch=%d\nfinished_epoch=%d\ncap_seconds=180\n' "$rc" "$cleanup_rc" "$start" "$done" > "$out/status.txt"
  (cd "$out" && find . -maxdepth 1 -type f ! -name observations.sha256 -print0 | sort -z | xargs -0 sha256sum) > "$out/observations.sha256"
  (( cleanup_rc == 0 )) || exit 125
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT

admitted=0
for attempt in 1 2 3; do
  if night_admit_idle >> "$out/admission.txt" 2>&1; then admitted=1; break; fi
  if (( attempt < 3 )); then sleep 3; fi
done
(( admitted == 1 )) || exit 90
cp FROZEN.sha256 INPUTS.sha256 build.json RELEASE.md "$out"
printf '%s\n' "${flags[@]}" > "$out/flags.txt"

run() {
  local label=$1 selected=$2 library=$3 left
  left=$((deadline - $(date +%s) - 20))
  (( left > 0 )) || return 124
  current=nmmoon-r2stride-$label-$id
  timeout -k 2s "${left}s" docker run --pull never --name "$current" --label "q2r2stride.owner=$id" \
    --network none --cpuset-cpus 2-3 --device "nvidia.com/gpu=$N3_GPU" \
    -v "$runtime:/candidate:ro" -v "$card/$library:/candidate/libggml-cuda.so:ro" \
    -v "$card/test-backend-ops:/candidate/test-backend-ops:ro" -v "$out:/evidence" \
    --entrypoint /bin/sh "${flags[@]}" \
    -e GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16=1 -e "GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4=$selected" -e "GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE=$selected" -e "GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16_CHECK=$selected" \
    -e "OCCAMY_Q2_R2_OUTPUT_PATH=/evidence/$label.bin" "$night_image" -ec '
      sha256sum /candidate/libggml-cuda.so /candidate/test-backend-ops
      exec /candidate/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT -p "kv=81920,nb=2,.*mask=mrow_causal,tail=260" -j 1
    ' "r2-$label" > "$out/$label.log" 2>&1
  clean
}

run a-control 0 control-relinked.so
run a-candidate 1 libggml-cuda.so
run b-candidate 1 libggml-cuda.so
run b-control 0 control-relinked.so
python3 -B report.py "$out" | tee "$out/analysis.log"
