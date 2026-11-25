#!/bin/bash
# card-2 S2 chain: wait for the S1 process to exit (it holds the lease), then BFCL preview native ckpt 2 and ckpt 8 back-to-back.
cd /tmp/qa-prefill-card || exit 1
[ -n "$1" ] && tail --pid="$1" -f /dev/null
for C in 2 8; do
  echo "$(date '+%F %T') S2 start ckpt=$C" >> s2.out
  QA_CKPT=$C QA_SUB=s2 env -u GGML_CUDA_OCCAMY_DEBUG -u GGML_CUDA_MAXWELL_GRAPHS python3 -I scripts/s2_bfcl.py preview native ckpt$C >> s2.out 2>&1
  echo "$(date '+%F %T') S2 end ckpt=$C rc=$?" >> s2.out
done
