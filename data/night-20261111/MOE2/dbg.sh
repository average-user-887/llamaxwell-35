#!/bin/bash
# MOE2 debug: TAILBAL kernel with main loop over all rows + tail recompute overwriting rows >= 1920 (lib-dbg1). Lock held externally.
R=/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/MOE2
OFF="GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_MOE_LOP3=1"
echo -n "D1 "; $R/g.sh C24 /tmp/moe1/lib-dbg1.so g1-D1 64 2048 $OFF GGML_CUDA_MOE_TAILBAL=1
$R/cmp.sh g1-A1 g1-D1
