#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
sha256sum /tool/state /candidate/libllama.so /candidate/libggml.so /candidate/libggml-base.so /candidate/libggml-cuda.so
for mode in wide_a wide_b serial; do
  /tool/state "$mode" /models/occamy-1.0-Q4_0-headq40b-mtp.gguf /fixture/prompt.ids /fixture/oracle.ids /fixture/prefill_ends.ids /out > "/out/$mode.rows" 2> "/out/$mode.log"
  for stage in prefix badquad; do
    if [[ $mode == serial && $stage == badquad ]]; then continue; fi
    for kind in state kv; do
      name=$mode.$stage.$kind
      (cd /out && sha256sum "$name") > "/out/$name.sha256"
      rm "/out/$name"
    done
  done
done
