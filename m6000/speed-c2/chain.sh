#!/usr/bin/env bash
cd /tmp/qa-speed-cards/c2
S=$(date -d "today 08:15" +%s)
E="env -u GGML_CUDA_OCCAMY_DEBUG -u GGML_CUDA_MAXWELL_GRAPHS -u N3_GPU QA_STOP_EPOCH=$S"
$E QA_ORDER=OFF84,ON84,ON84,OFF84,ON64,OFF64,OFF64,ON64 QA_OUT=/tmp/qa-speed-cards/c2/runs python3 -B scripts/c2_abba.py >> timing.out 2>&1
$E QA_MV2=1 QA_ORDER=OFF64,ON64 QA_OUT=/tmp/qa-speed-cards/c2/mv2 python3 -B scripts/c2_abba.py >> mv2.out 2>&1
echo CHAIN_DONE
