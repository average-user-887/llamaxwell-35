#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
card=$(cd -- "$(dirname -- "$0")" && pwd -P)
campaign=$(dirname "$card")
lab=$(dirname "$campaign")
runner=$campaign/runner
source "$runner/night_trial_runtime.sh"
libs=$campaign/night-native-depth2-q3-80k-r1/libs
input=$campaign/night-gdn-t2-state/inputs
n1=$campaign/night-cache-only/libs
r2=$campaign/sparse/r2/libs
model_dir=$(dirname "$lab")/uroboros-models
model=$model_dir/occamy-1.0-Q4_0-headq40b-mtp.gguf
profile=$campaign/night-native-depth2-q3-80k-r1/receipts/@PRIVATE_BASELINE_RUN_NAME@/environment.args
cd "$card"
[[ $# == 1 && ( $1 == --check || $1 == --run ) ]] || exit 2
sha256sum -c --quiet FROZEN.sha256
[[ $(stat -Lc '%s %Y' "$model") == '20080296992 1791834438' ]]
grep -Fxq "@PRIVATE_MODEL_FILE_SHA256@  $model" "$campaign/model.sha256"
LD_LIBRARY_PATH="$n1" ./state --self-test
python3 -B offline.py
if [[ $1 == --check ]]; then exit 0; fi
night_lock_gpu
begin=$(date +%s)
deadline=$((begin+190))
if [[ -n ${ROOT_GPU_DEADLINE_EPOCH:-} ]]; then [[ $ROOT_GPU_DEADLINE_EPOCH =~ ^[0-9]+$ ]] && ((deadline<=ROOT_GPU_DEADLINE_EPOCH)) || exit 124; fi
id=$(date -u +%Y%m%dT%H%M%SZ)-$$
out=$card/receipts/$id
name=nmmoon-resume-roll4-$id
mkdir -p "$out/raw"
chmod 0777 "$out/raw"
printf 'ROLL4_RECEIPT=%s\n' "$out"
cp FROZEN.sha256 "$out"
cp "$profile" "$out/environment.args"
printf '%s\n' -e GGML_CUDA_OCCAMY_FA_Q4_R2=1 -e GGML_CUDA_OCCAMY_FA_Q4_R2_NU=2 -e GGML_CUDA_OCCAMY_FA_Q4_R2_CHECK=0 -e GGML_CUDA_OCCAMY_MOE_Q3_COHORTS=0 -e GGML_CUDA_OCCAMY_MOE_Q4_COHORTS=1 >> "$out/environment.args"
printf 'image=%s\ncap_seconds=190\nstart_epoch=%s\nallocation=one_owned_target_only_fixture\n' "$night_image" "$begin" > "$out/manifest.txt"
cleanup() {
  local owned
  owned=$(timeout -k 1 3 docker ps -aq --filter "name=^/$name$" --filter "label=roll4.owner=$id") || return 1
  if [[ -n $owned ]]; then [[ $owned =~ ^[a-f0-9]+$ ]] && timeout -k 1 5 docker rm -f "$owned" || return 1; fi
  [[ -z $(timeout -k 1 3 docker ps -aq --filter "name=^/$name$") ]]
}
finish() {
  local rc=$? cleaned=0
  trap - EXIT TERM INT
  cleanup > "$out/cleanup.log" 2>&1 || cleaned=$?
  (( $(date +%s) <= deadline )) || rc=124
  printf 'runner_exit=%s\ncleanup_exit=%s\nend_epoch=%s\n' "$rc" "$cleaned" "$(date +%s)" > "$out/status.txt"
  python3 -B "$card/safe-report.py" "$out" || rc=2
  (cd "$out"; find . -type f ! -name observations.sha256 -print0 | sort -z | xargs -0 sha256sum) > "$out/observations.sha256"
  ((cleaned==0)) || exit 125
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' TERM INT
(( $(df -PB1 "$card" | awk 'NR==2{print $4}') >= 1153433600 )) || { echo 'insufficient archive space' > "$out/error.txt"; exit 1; }
night_admit_idle > "$out/admission.txt"
mapfile -t args < "$out/environment.args"
left=$((deadline-$(date +%s)-28)); ((left>0)) || exit 124
timeout -k 2 "${left}s" docker run --pull never --name "$name" --label "roll4.owner=$id" --network none \
  --device "nvidia.com/gpu=$N3_GPU" --cpuset-cpus 2-3 "${args[@]}" -v "$libs:/candidate:ro" -v "$campaign/resume-q4-combined-native/libggml-cuda.so:/candidate/libggml-cuda.so:ro" \
  -v "$n1:$n1:ro" -v "$r2:$r2:ro" -v "$model_dir:/models:ro" -v "$input:/fixture:ro" \
  -v "$card/state:/tool/state:ro" -v "$card/in-container.sh:/tool/run:ro" -v "$out/raw:/out" \
  --entrypoint /bin/bash "$night_image" -c 'exec nice -n 19 /bin/bash /tool/run' > "$out/raw/container.log" 2>&1
cleanup >> "$out/cleanup.log" 2>&1
left=$((deadline-$(date +%s)-15)); ((left>0)) || exit 124
timeout -k 1 "${left}s" taskset -c 2-3 nice -n 19 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B analyze.py "$out/raw" 2>&1 | tee "$out/analysis.log"
cp "$out/raw/result.json" "$out/result.json"
