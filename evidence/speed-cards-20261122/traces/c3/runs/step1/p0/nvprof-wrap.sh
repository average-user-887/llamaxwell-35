#!/bin/bash
exec /usr/local/cuda/bin/nvprof --print-gpu-trace --csv --normalized-time-unit us --log-file /out/nvprof.csv /real/llama-server "$@"
