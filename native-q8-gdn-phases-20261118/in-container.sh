#!/usr/bin/env bash
set -euo pipefail
umask 022
export GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OCCAMY_GDN_T2_Q8A=1 GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=0 GGML_CUDA_OCCAMY_MOE_Q2_COHORTS=1 GGML_CUDA_OCCAMY_MOE_Q2_COUNT=1
for arm in control clock; do
    export GGML_CUDA_OCCAMY_Q8_PHASE_CLOCK=$arm
    (
      cd "/out/$arm"
      printf "Q8CLK_DSO " > t2.log
      sha256sum /candidate/libggml-cuda.so >> t2.log
      /tool/state t2 /models/occamy-1.0-Q4_0-headq40b-mtp.gguf /fixture/prompt.ids /fixture/oracle.ids \
          /fixture/prefill_ends.ids "/out/$arm" > t2.rows \
          2> >(sed -uE 's/GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}/GPU-REDACTED/gI' >> t2.log)
      files=(t2.prefix.state t2.badpair.state t2.kept.state t2.corrected.state t2.badpair.logits t2.corrected.logits)
      sha256sum "${files[@]}" > outputs.sha256
      for f in "${files[@]}"; do stat -c '%s %n' "$f"; done > outputs.bytes
      rm -- "${files[@]}"
    )
    echo "Q8CLK_FINISHED arm=$arm"
done
