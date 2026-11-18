from pathlib import Path
old=Path('../resume-q8-on-serial/release.sh').read_text()
s=old.replace('probe=$card/serial_probe.py','probe=$card/model.py').replace('n82=$campaign/night-gdn-t2-q8-fast','n82=$campaign/resume-q2-moe-cohorts')
s=s.replace('resume.q8.serial.owner','resume.q2.model.owner').replace('RESUME_Q8_SERIAL_LOCK_FD','RESUME_Q2_MODEL_LOCK_FD').replace('nmmoon-resume-q8-serial-', 'nmmoon-resume-q2-model-').replace('RESUME_Q8_SERIAL_RECEIPT','Q2_MODEL_RECEIPT').replace('Q8_SERIAL_RECEIPT','Q2_MODEL_RECEIPT')
s=s.replace('    [[ $2 == serial ]]','    [[ $2 == serial || $2 == native ]]').replace('&& ( $2 == serial )','&& ( $2 == serial || $2 == native )').replace('--run serial | --compare SERIAL_RECEIPT','--run native|serial | --compare SERIAL_RECEIPT NATIVE_RECEIPT').replace('  --compare) [[ $# == 2 ]]','  --compare) [[ $# == 3 ]]').replace('compare "$2" ;;','compare "$2" "$3" ;;')
s=s.replace('  python3 -B "$probe" check','  python3 -B "$card/check-layout.py"\n  python3 -B "$probe" check')
a=s.index('  local name=$1 out=$2 path inode');b=s.index('\n}\n',a)
s=s[:a]+'''  local name=$1 out=$2 child=$3 path inode
  bounded_docker exec "$name" cat "/proc/$child/maps" |
    awk '$6 ~ /\\/lib(ggml-cuda\\.so|llama-server-impl\\.so|llama(-common)?\\.so(\\.[0-9]+)*)$/ {
      if (NF != 6) exit 2; print $6, $5
    }' | sort -u > "$out/runtime-library-inodes.txt"
  {
    while read -r path inode; do
      [[ $(bounded_docker exec "$name" stat -Lc '%i' "$path") == "$inode" ]]
      bounded_docker exec "$name" sha256sum "$path"
    done < "$out/runtime-library-inodes.txt"
    bounded_docker exec "$name" sha256sum "/proc/$child/exe"
  } > "$out/runtime-library-hashes.txt"
  bounded_docker exec "$name" sha256sum /proc/1/exe > "$out/supervisor.sha256"'''+s[b:]
a=s.index('arm() {')
s=s[:a]+'''sample() {
  local name=$1 out=$2 phase=$3
  [[ ! -e $out/witness/$phase.json ]]
  bounded_docker kill --signal=USR1 "$name" >/dev/null
  for _ in {1..50}; do
    [[ -s $out/witness/$phase.json ]] && return 0
    sleep .1
  done
  echo "No owned-child counter sample for $phase" >&2
  return 2
}

'''+s[a:]
s=s.replace('runtime_inventory "$name" "$out"','''local child
  child=$(cat "$out/witness/child.pid")
  [[ $child =~ ^[0-9]+$ && $child -gt 1 ]]
  runtime_inventory "$name" "$out" "$child"''')
s=s.replace('  bounded_docker run -d','  mkdir "$out/witness"\n  bounded_docker run -d').replace('--log-opt max-size=4m','--log-opt max-size=24m')
s=s.replace('--entrypoint /candidate/llama-server','-v "$card/supervise:/supervise:ro" -v "$out/witness:/witness:rw" --entrypoint /supervise')
s=s.replace("bounded_docker exec \"$name\" sh -ec 'tr \"\\000\" \"\\n\" < /proc/1/environ | grep -E \"$1\"' flags \"$flag_pattern\" > \"$out/pid1-selected-environment.txt\"",'''bounded_docker exec "$name" sh -ec 'tr "\\000" "\\n" < "/proc/$2/environ" | grep -E "$1"' flags "$flag_pattern" "$child" > "$out/child-selected-environment.txt"''')
s=s.replace("bounded_docker exec \"$name\" sh -ec 'tr \"\\000\" \"\\n\" < /proc/1/cmdline' > \"$out/pid1-server.args\"",'''bounded_docker exec "$name" sh -ec 'tr "\\000" "\\n" < "/proc/$1/cmdline"' args "$child" > "$out/child-server.args"''')
s=s.replace('python3 -B "$probe" ready "$out"','python3 -B "$probe" ready "$stage" "$out"\n  sample "$name" "$out" start')
s=s.replace('  capture_logs "$name" "$out/server-pre-warm.log"','  sample "$name" "$out" cold\n  capture_logs "$name" "$out/server-pre-warm.log"')
s=s.replace('  bounded_docker stop --time 5','  sample "$name" "$out" warm\n  bounded_docker stop --time 5')
s=s.replace('python3 -B "$probe" verify "$out"','python3 -B "$probe" verify "$stage" "$out"').replace('python3 -B "$probe" finalize "$out"','python3 -B "$probe" finalize "$stage" "$out"').replace('RQS_RECEIPT=', 'Q2_MODEL_RECEIPT=')
assert '/proc/1/environ' not in s and 'pid1-server.args' not in s
Path('release.sh').write_text(s)
