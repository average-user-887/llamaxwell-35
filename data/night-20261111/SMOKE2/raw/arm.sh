#!/bin/bash
# usage: arm.sh A|B
ARM=$1; M=/media/avg-usr/Storage/m6000-llama-cpp; N=$M/occamy35-night-20261111; R=$N/rounds/SMOKE2/raw; PORT=18772
APP=$N/app
ENVS=(GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_OCCAMY_FA=1 GGML_CUDA_OCCAMY_FUSED=1 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_MOE_DOWNX=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1)
LIB=$M/prod-libs/libggml-cuda-call35-20261111.so
if [ $ARM = B ]; then LIB=$N/rounds/BUNDLE2/libggml-cuda.bundle2.so; ENVS+=(GGML_CUDA_OCCAMY_FA_PIPE=3 GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2); fi
EA=(); for e in "${ENVS[@]}"; do EA+=(-e $e); done
NAME=smoke2-serve-$ARM
docker run -d --name $NAME -p 127.0.0.1:$PORT:8080 --device nvidia.com/gpu=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e \
 -v /media/avg-usr/Storage/uroboros-models:/models:ro -v $APP:/app:ro -v $LIB:/app/libggml-cuda.so:ro -e LD_LIBRARY_PATH=/app "${EA[@]}" \
 --entrypoint /app/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 --cache-ram 12288 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics -lv 4 >/dev/null
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 3; done
curl -sf http://127.0.0.1:$PORT/health || { echo HEALTH_FAIL; docker logs $NAME 2>&1 | tail -20; docker rm -f $NAME; exit 1; }
nvidia-smi -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > $R/vram-$ARM.txt & SM=$!
python3 -I $R/drive.py $PORT $R/$ARM.json
rc=$?
python3 -I $R/diag.py $PORT $R/diag-$ARM.json
kill $SM
docker logs $NAME > $R/server-$ARM.log 2>&1
docker rm -f $NAME >/dev/null
exit $rc
