#!/bin/bash
# waits for the S2 chain, then runs S3 logprob parity arms under one lease
tail --pid=$1 -f /dev/null
cd /tmp && python3 -I /tmp/qa-prefill-card/scripts/s3_lp.py A1:1024:2 A2:1024:2 B:2048:2:GGML_CUDA_FA_MAXWELL_GEMM_V5=1 C8:1024:8 >> /tmp/qa-prefill-card/s3.out 2>&1
echo "s3 rc=$?" >> /tmp/qa-prefill-card/s3.out
