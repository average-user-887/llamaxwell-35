#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
card=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
campaign=$(dirname -- "$card")
lab=$(dirname -- "$campaign")
runner=$campaign/runner
source "$runner/night_trial_runtime.sh"
base=$campaign/night-gdn-t2-state
libs=$campaign/night-cache-only/libs
models=$(dirname -- "$lab")/uroboros-models
model=occamy-1.0-Q4_0-headq40b-mtp.gguf
check() {
    (cd "$card"; sha256sum --quiet -c PACKET.sha256; sha256sum --quiet -c INPUTS.sha256)
    (cd "$libs"; sha256sum --quiet -c "$base/RUNTIME.sha256")
    night_profile
    extensions=(-e LD_LIBRARY_PATH=/candidate -e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4)
    cmp -s "$base"/receipts/nmmoon-n34-2k-20261118T013121Z-2517339-8668/t2.environment.args <(printf '%s\n' "${night_cuda[@]}" "${extensions[@]}")
    [[ $(stat -c '%s %Y' "$models/$model") == '20080296992 1791834438' ]]
    rg -Fqx -- "d5ff985eea0bcb3a7305a2033582bf63f756c9f4b3700d96b4939f81ee041eae  $models/$model" "$campaign/model.sha256"
    python3 -B "$card/check-analyzer.py"
    (cd "$card"; python3 -B verify_sass.py | cmp -s - compiled.json)
}
case ${1-} in
    --check) [[ $# == 1 ]]; check; echo Q8CLK_OFFLINE_OK; exit 0 ;;
    2k) [[ $# == 1 ]] ;;
    *) echo 'Use --check or 2k' >&2; exit 2 ;;
esac
start=$(date +%s)
night_cutoff=$(TZ=Europe/Warsaw date -d "2026-11-18 17:00:00" +%s)
if [[ -n ${ROOT_GPU_DEADLINE_EPOCH:-} ]]; then [[ $ROOT_GPU_DEADLINE_EPOCH =~ ^[0-9]+$ ]]; (( ROOT_GPU_DEADLINE_EPOCH >= night_cutoff )) || night_cutoff=$ROOT_GPU_DEADLINE_EPOCH; fi
deadline=$((start+180))
((deadline <= night_cutoff)) || { echo '180s including cleanup would exceed cutoff'; exit 2; }
check
used=$(du -sb "$card" | cut -f1)
((used + 300*1024*1024 < 1024*1024*1024))
(( $(df -PB1 "$card" | awk 'NR==2{print $4}') >= 300*1024*1024 ))
night_lock_gpu
id=$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM
name="nmmoon-q8clk-2k-$id"
out=$card/receipts/$name
mkdir -p "$out/control" "$out/clock"
printf 'Q8CLK_RECEIPT=%s\n' "$out"
printf 'stage=2k\ncap_s=180\nstart_epoch=%d\ndeadline_epoch=%d\n' "$start" "$deadline" > "$out/status.txt"
sha256sum "$card/PACKET.sha256" "$card/INPUTS.sha256" > "$out/packet.sha256"
printf '%s\n' "${night_cuda[@]}" "${extensions[@]}" -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_OCCAMY_GDN_T2_Q8A=1 -e GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0 -e GGML_CUDA_OCCAMY_MOE_Q2_COHORTS=1 -e GGML_CUDA_OCCAMY_MOE_Q2_COUNT=1 > "$out/environment.base.args"
printf 'image=%s\nmodel=%s\nmodel_size=%s\nmodel_mtime=%s\n' "$night_image" "$model" "$(stat -c %s "$models/$model")" "$(stat -c %Y "$models/$model")" > "$out/manifest.txt"
night_admit_idle > "$out/admission.txt"
[[ -s $out/admission.txt ]]
label="q8clk.owner=$id"
cleanup() {
    local ids
    ids=$(timeout -k 1s 2s docker ps -aq --filter "name=^/$name$" --filter "label=$label") || return 1
    if [[ -n $ids ]]; then
        [[ $ids != *$'\n'* ]] || return 1
        timeout -k 1s 3s docker rm -f "$ids" >> "$out/cleanup.log" || return 1
    fi
    ids=$(timeout -k 1s 2s docker ps -aq --filter "name=^/$name$" --filter "label=$label") || return 1
    [[ -z $ids ]]
}
running=0
trap 'if ((running)); then cleanup || echo cleanup=unconfirmed >> "$out/status.txt"; fi' EXIT
trap 'night_interrupt; exit 130' INT TERM
left=$((deadline-$(date +%s)-18))
((left>0))
running=1
timeout --signal=TERM --kill-after=2s "${left}s" docker run --rm --pull never --network none \
    --name "$name" --label "$label" --log-driver none --device "nvidia.com/gpu=$N3_GPU" \
    "${night_cuda[@]}" "${extensions[@]}" \
    -v "$libs:/candidate:ro" -v "$card/libggml-cuda.so:/candidate/libggml-cuda.so:ro" \
    -v "$models:/models:ro" -v "$card/state:/tool/state:ro" -v "$base/inputs:/fixture:ro" \
    -v "$card/in-container.sh:/tool/run:ro" -v "$out:/out" \
    --entrypoint /bin/bash "$night_image" /tool/run > "$out/container.log" 2>&1 &
night_child_pid=$!
rc=0
wait "$night_child_pid" || rc=$?
night_child_pid=
cleanup || { echo cleanup=unconfirmed >> "$out/status.txt"; exit 125; }
running=0
printf 'runner_exit=%d\ncleanup=complete\ncleaned_epoch=%d\n' "$rc" "$(date +%s)" >> "$out/status.txt"
flock -u "$night_lock_fd"
(( rc == 0 )) || { echo status=incomplete >> "$out/status.txt"; exit "$rc"; }
left=$((deadline-$(date +%s)-1))
(( left>0 ))
timeout -k 1s "${left}s" taskset -c 0-1 nice -n 19 python3 -B "$card/analyze.py" "$out" > "$out/summary.json" || { echo status=incomplete >> "$out/status.txt"; exit 2; }
(cd "$out"; sha256sum control/* clock/* q8.clock.u32 summary.json > observations.sha256)
echo status=complete >> "$out/status.txt"
cat "$out/summary.json"
