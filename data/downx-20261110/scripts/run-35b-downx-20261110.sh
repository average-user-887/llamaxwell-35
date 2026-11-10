#!/bin/bash
# 35B moe (10 Nov 2026): ub1024 prod + GGML_CUDA_MOE_DOWNX=1 (exact, +1.1% tg, llamaxwell-35 moe-downx 897a09c). Adopt: docker stop llamacpp-m6000; docker rename llamacpp-m6000 llamacpp-m6000-ub1024-prev; docker rename llamacpp-m6000-downx llamacpp-m6000; (swap starts it). Rollback: reverse renames.
docker create --name llamacpp-m6000-downx --network ai-net --restart no -p 127.0.0.1:11436:8080 \
  --device nvidia.com/gpu=GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e \
  -v /media/avg-usr/Storage/ollama-m6000/gguf:/gguf -v /media/avg-usr/Storage/uroboros-models:/models -v /media/avg-usr/Storage/llama-slots-m6000:/slots \
  -v /media/avg-usr/Storage/m6000-llama-cpp/prod-libs/libggml-cuda-downx35-20261110.so:/app/libggml-cuda.so:ro \
  -e GGML_CUDA_MOE_DOWNX=1 -e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_RMS_WARP=0 -e GGML_CUDA_HEAD_ROWPACK_SM52=1 -e GGML_CUDA_MMQ_COMPACT_SM52=1 -e GGML_CUDA_Q8_REUSE=1 -e GGML_CUDA_NORM_Q8=1 -e GGML_CUDA_CONCAT_T=1 -e GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 -e GGML_CUDA_GATE_COPY=1 -e GGML_CUDA_OCCAMY_GDN=1 -e GGML_CUDA_MMQ_XMAD_SM52=1 -e GGML_CUDA_GDN_COLS=2,4 -e GGML_CUDA_OCCAMY_FA=1 -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MMQ_TAIL_SM52=1 \
  sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
  -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 1024 --cache-ram 12288 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics --slot-save-path /slots -lv 4
