#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

card=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
campaign=$(dirname -- "$card")
lab=$(dirname -- "$campaign")
runner=$campaign/runner
source "$runner/night_trial_runtime.sh"
runtime=$campaign/sparse/r2/libs
n88=$campaign/night-gdn-q8-long/receipts/on-20261118T041311Z-2977561-30828
cap=180
cd -- "$card"

case ${1:-} in
  --check|--run) [[ $# == 1 ]] || exit 2 ;;
  *) echo 'usage: release.sh --check | --run' >&2; exit 2 ;;
esac

sha256sum --check --quiet FROZEN.sha256
sha256sum --check --quiet SOURCES.sha256
sha256sum --check --quiet RUNTIME.sha256
sha256sum --check --quiet INPUTS.sha256
LD_LIBRARY_PATH="$runtime" ./operator --fixture 2>&1 | cmp fixture.txt -
python3 -B report.py --check

# Select only public, relevant CUDA flags from the actual N88 Q8-enabled arm.
mapfile -t flags < <(grep -E '^GGML_CUDA_(OCCAMY_GDN_T2_Q8A(_EVENTS)?|MOE_(DOWNX|DOWNH|LOP3|TAILBAL|V3)|OCCAMY_(FUSED|N2SPLIT|HOSTGAP|SMALLFUSE)|MMVQ_(SM52|PIPE))=([[:digit:]])+$' "$n88/environment.args")
[[ ${#flags[@]} -ge 8 ]]
for required in GGML_CUDA_OCCAMY_GDN_T2_Q8A=1 GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0 GGML_CUDA_OCCAMY_N2SPLIT=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_MMVQ_SM52=1; do
  printf '%s\n' "${flags[@]}" | grep -Fxq "$required"
done
if [[ $1 == --check ]]; then
  echo 'Exact paired four-row down Q2 operator gates verified offline.'
  exit 0
fi

night_lock_gpu
start=$(date +%s)
deadline=$((start + cap))
: "${ROOT_GPU_DEADLINE_EPOCH:?root operator lease end required}"
if [[ -n ${ROOT_GPU_DEADLINE_EPOCH:-} ]]; then
  [[ $ROOT_GPU_DEADLINE_EPOCH =~ ^[0-9]+$ ]] || exit 2
  (( deadline <= ROOT_GPU_DEADLINE_EPOCH )) || { echo '180-second operator stage extends beyond the supplied lease'; exit 124; }
fi
id=$(date -u +%Y%m%dT%H%M%SZ)-$$
out=$card/receipts/$id
mkdir -p -- "$out"
printf 'Q2_DOWN4_RECEIPT=%s\n' "$out"
current=

clean() {
  local ids
  [[ -n $current ]] || return 0
  ids=$(timeout -k 1s 3s docker ps -aq --filter "name=^/$current$" --filter "label=q2down4.owner=$id") || return 1
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
  if [[ -n ${clock_pid:-} ]]; then touch "$out/perf.done"; wait "$clock_pid" || true; fi
  clean || cleanup_rc=$?
  done=$(date +%s)
  (( done <= deadline )) || rc=124
  printf 'runner_exit=%d\ncleanup_exit=%d\nstarted_epoch=%d\nfinished_epoch=%d\ncap_seconds=%d\n' \
    "$rc" "$cleanup_rc" "$start" "$done" "$cap" > "$out/status.txt"
  python3 -B "$card/report.py" --export "$out" || rc=2
  (cd "$out" && find . -type f ! -name observations.sha256 -print0 | sort -z | xargs -0 sha256sum) > "$out/observations.sha256"
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

cp FROZEN.sha256 RUNTIME.sha256 build.json RELEASE.md "$out"
printf '%s\n' "${flags[@]}" > "$out/flags.txt"

run() {
  local mode=$1 left debug=0 checks=0 item
  local -a env=(-e LD_LIBRARY_PATH=/runtime -e GGML_CUDA_DISABLE_GRAPHS=1)
  left=$((deadline - $(date +%s) - 20))
  (( left > 0 )) || return 124
  if [[ $mode == check ]]; then debug=1; checks=0; fi
  for item in "${flags[@]}"; do env+=(-e "$item"); done
  env+=(-e "GGML_CUDA_OCCAMY_DEBUG=$debug" -e "GGML_CUDA_MOE_V3_CHECK=$checks")
  current=nmmoon-q2down4-$mode-$id
  timeout -k 2s "${left}s" docker run --pull never --name "$current" --label "q2down4.owner=$id" \
    --network none --device "nvidia.com/gpu=$N3_GPU" --cpuset-cpus 2-3 \
    -v "$runtime:/runtime:ro" -v "$card/libggml-cuda.so:/runtime/libggml-cuda.so:ro" \
    -v "$card/operator:/operator:ro" --entrypoint /bin/sh "${env[@]}" "$night_image" -ec '
      sha256sum /runtime/libggml.so.0.26.0 /runtime/libggml-base.so.0.26.0 /runtime/libggml-cuda.so /operator
      exec nice -n 19 /operator "$1"
    ' q2down4 "--$mode" > "$out/$mode.log" 2>&1
  clean
}

run check
python3 -B report.py --numeric "$out" 2>&1 | tee "$out/numeric-validation.log"
python3 -B "$card/fleet-clock.py" --sample "$out" "$deadline" &
clock_pid=$!
run perf
touch "$out/perf.done"
wait "$clock_pid"
python3 -B report.py --report "$out" 2>&1 | tee "$out/analysis.log"
