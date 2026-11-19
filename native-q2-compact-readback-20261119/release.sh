#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
card=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
campaign=$(dirname -- "$card")
lab=$(dirname -- "$campaign")
runner=$campaign/runner
source "$runner/night_trial_runtime.sh"
runtime=$campaign/night-one-graph-draft/build/bin
cd -- "$card"
[[ $# == 1 && ($1 == --check || $1 == --run) ]] || exit 2
sha256sum --check --quiet FROZEN.sha256
sha256sum --check --quiet INPUTS.sha256
if [[ $1 == --check ]]; then
  echo 'Compact readback frozen source, sm_52 executable and pinned CPU runtime checks passed.'
  exit
fi

night_lock_gpu
start=$(date +%s)
deadline=$((start + 120))
if [[ -n ${ROOT_GPU_DEADLINE_EPOCH:-} ]]; then
  [[ $ROOT_GPU_DEADLINE_EPOCH =~ ^[0-9]+$ ]] || exit 2
  (( deadline <= ROOT_GPU_DEADLINE_EPOCH )) || exit 124
fi
id=$(date -u +%Y%m%dT%H%M%SZ)-$$
out=$card/receipts/$id
name=nmmoon-compact-readback-$id
mkdir -p -- "$out"
printf 'COMPACT_READBACK_RECEIPT=%s\n' "$out"

finish() {
  local status=$? cleanup=0 remaining done
  trap - EXIT INT TERM
  remaining=$(timeout -k 1s 3s docker ps -aq --filter "name=^/$name$" --filter "label=compact.readback.owner=$id") || cleanup=1
  if [[ -n $remaining ]]; then
    if [[ $remaining =~ ^[0-9a-f]+$ ]]; then
      timeout -k 1s 5s docker rm -f "$remaining" >> "$out/cleanup.log" 2>&1 || cleanup=1
    else
      cleanup=1
    fi
  fi
  remaining=$(timeout -k 1s 3s docker ps -aq --filter "name=^/$name$") || cleanup=1
  [[ -z $remaining ]] || cleanup=1
  done=$(date +%s)
  (( done <= deadline )) || status=124
  printf 'runner_exit=%d\ncleanup_exit=%d\nstarted_epoch=%d\nfinished_epoch=%d\ncap_seconds=120\n' \
    "$status" "$cleanup" "$start" "$done" > "$out/status.txt"
  (cd "$out" && find . -maxdepth 1 -type f ! -name observations.sha256 -print0 | sort -z | xargs -0 sha256sum) > "$out/observations.sha256"
  (( cleanup == 0 )) || exit 125
  exit "$status"
}
trap finish EXIT
trap 'exit 124' INT TERM

night_admit_idle > "$out/admission.txt" 2>&1
cp FROZEN.sha256 INPUTS.sha256 README.md "$out"
left=$((deadline - $(date +%s) - 20))
(( left > 0 )) || exit 124
timeout -k 2s "${left}s" docker run --pull never --name "$name" --label "compact.readback.owner=$id" \
  --network none --device "nvidia.com/gpu=$N3_GPU" --cpuset-cpus 0-1 \
  -v "$runtime:/runtime:ro" -v "$card:/candidate:ro" -e LD_LIBRARY_PATH=/runtime \
  --entrypoint /bin/sh "$night_image" -ec '
    sha256sum /runtime/libllama-common.so.0.6.0 /runtime/libllama.so.0.6.0 /candidate/compact-readback
    exec nice -n 19 /candidate/compact-readback --gpu
  ' > "$out/operator.log" 2>&1
python3 -B report.py "$out/operator.log" > "$out/result.json"
night_admit_idle > "$out/admission-post.txt" 2>&1 || true
