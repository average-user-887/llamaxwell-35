#!/bin/bash
# S2 rerun with thermal pacing (pause own container >=77 C until <=72 C / 15 s), after S3 chain
tail --pid=$1 -f /dev/null
cd /tmp
for C in 2 8; do
  echo "$(date '+%F %T') S2b start ckpt=$C" >> /tmp/qa-prefill-card/s2b.out
  QA_CKPT=$C QA_SUB=s2b python3 -I /tmp/qa-prefill-card/scripts/s2_bfcl.py preview native ckpt${C}p >> /tmp/qa-prefill-card/s2b.out 2>&1
  echo "$(date '+%F %T') S2b end ckpt=$C rc=$?" >> /tmp/qa-prefill-card/s2b.out
done
