#!/bin/bash
# srv.sh quick|full : bench llama-server (prod image, prod args, own port, own slot dir) with SRVLIB/SRVDX, run client, stop.
set -u
J=/home/avg-usr/.claude/jobs/fc533454/tmp; C5=$J/c5; O=${O:-$C5/out}; GPU=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e
SIMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
E=(); for a in GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 \
 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 \
 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_DEBUG=1; do E+=(-e "$a"); done
mkdir -p $O/slots/$NAME
docker rm -f c5-srv >/dev/null 2>&1
docker run -d --name c5-srv -p 127.0.0.1:${SRV_PORT:?}:8080 --device nvidia.com/gpu=$GPU -v /media/avg-usr/Storage/uroboros-models:/models:ro \
  -v $O/slots/$NAME:/slots -v $SRVLIB:/app/libggml-cuda.so:ro "${E[@]}" ${SRVX:-} -e GGML_CUDA_MOE_DOWNX=$SRVDX $SIMG \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 \
  --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots >/dev/null
for i in $(seq 1 120); do curl -sf 127.0.0.1:${SRV_PORT:?}/health >/dev/null && break; sleep 2; done
python3 -I $C5/scripts/dclient.py $1 > $O/$NAME.client.txt 2>&1; echo "client rc=$?" >> $O/$NAME.client.txt
docker stop -t 30 c5-srv >/dev/null; docker logs c5-srv > $O/$NAME.srv.log 2>&1; docker rm c5-srv >/dev/null
