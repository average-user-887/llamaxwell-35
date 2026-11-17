#!/bin/bash
# Quality-gate chain. Run: flock -w 600 gpu-bench.lock qgate_gpu.sh   (lock held by caller)
set -u
unset GGML_CUDA_MAXWELL_GRAPHS
M=<WORKDIR>; E=$M/mtp-verifier-20261117; Q=$E/qgate; GPU=1; PORT=18331
OUT=$Q/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $Q/LATEST
BASE=$E/run-c5-20261117T135150Z/p_A80/result.json
LIBS=<LIBS_DIR>
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nqg-' | xargs -r docker rm -f >/dev/null 2>&1; S cleanup; }
trap cleanup EXIT
curl -s -m60 -X POST <SWAP_PROXY>/unload/<model> > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
sha256sum $LIBS/libggml-cuda.so $LIBS/libllama-server-impl.so $LIBS/llama-server > $OUT/build.sha256
arm() { tag=$1; kind=$2; xe=$3
  n=nqg-$tag; D=$OUT/$tag; mkdir -p $D; chmod 777 $D; S "$tag start"
  FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2"
  [ $kind = B ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16"
  docker run -d --name $n --device nvidia.com/gpu=$GPU -p 127.0.0.1:$PORT:8080 -v <MODELS_DIR>:/models:ro -v $LIBS:/candidate:ro -v $D:/mv1out \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS -e LLAMA_MV2_DUMP=/mv1out/dump.txt -e LLAMA_MV2_DUMP_LO=0 -e LLAMA_MV2_DUMP_HI=64 $xe $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 1 -c 84224 -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0; for i in $(seq 1 240); do curl -sf -m3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }; [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2; done
  [ $ok = 1 ] && python3 -I $Q/qgate_client.py 127.0.0.1:$PORT $D/result.json $D/dump.txt $BASE > $D/client.out 2>&1
  docker stop -t 60 $n >/dev/null 2>&1; docker logs $n > $D/server.log 2>&1; docker rm $n >/dev/null 2>&1
  for j in $(seq 1 60); do [ "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok"; }
BE="-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4 -e LLAMA_MTP_MERGE_REEVAL=1"
arm A A ""
arm Bfull B "$BE"
arm B_fa_mrow1 B "-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=1 -e LLAMA_MTP_MERGE_REEVAL=1"
arm B_fa_stock B "-e GGML_CUDA_OCCAMY_GDN_T2=1 -e LLAMA_MTP_MERGE_REEVAL=1"
arm B_no_t2 B "-e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4 -e LLAMA_MTP_MERGE_REEVAL=1"
arm B_no_merge B "-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4"
arm B_stock B ""
S "all done"
