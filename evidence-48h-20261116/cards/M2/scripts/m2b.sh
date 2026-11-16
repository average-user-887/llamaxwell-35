#!/bin/bash
# Card M2b: MTP long-context decode/prefill matrix. Env: N3_GPU (uuid), N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *unload/qwen36-35b-moe) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/M2
OUT=$E/runb-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATESTB
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nm2-' | xargs -r docker rm -f >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"; T0=$(date +%s)
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
Q="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
arm() { # tag np perslot spec(0|n) maxdepth extra_env
  tag=$1; np=$2; ps=$3; sp=$4; md=$5; xe=$6; n=nm2-$tag; D=$OUT/$tag; mkdir -p $D
  if [ $(( $(date +%s) - T0 )) -gt 999999 ]; then S "$tag skipped (time budget)"; return; fi
  S "$tag start np=$np per_slot=$ps spec=$sp"
  if [ "$tag" = prod ]; then
    MODEL=/models/occamy-1.0-Q4_0-headq40b.gguf; FL="-ub 2048 --cache-ram 12288 --ctx-checkpoints 16"
  else
    MODEL=/models/occamy-1.0-Q4_0-headq40b-mtp.gguf; FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2"
    [ "$sp" != 0 ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max $sp $Q"
  fi
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p 127.0.0.1:11476:8080 \
    -v <MEDIA>/Storage/uroboros-models:/models:ro -v $M/prod-libs/batch8-20261115:/candidate:ro \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $xe $IMG \
    -m $MODEL --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np $np -c $((ps*np)) -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://127.0.0.1:11476/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    python3 -I $E/m2_client.py 11476 $tag $np $md $D/result.json > $D/client.out 2>&1
    curl -s -m10 http://127.0.0.1:11476/metrics > $D/metrics.txt
  fi
  docker logs $n > $D/server.log 2>&1
  grep -a -i "occamy_\|fallback\|spec" $D/server.log | grep -a -v "^srv.*params" | cut -c1-240 | sort | uniq -c | sort -rn | head -60 > $D/counters.txt
  kill $MON 2>/dev/null; docker rm -f $n >/dev/null 2>&1; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok peak=$(sort -n $D/vram.csv | tail -1)"
}
arm n3_np2 2 84224 3 80000 ""
arm dbg_n3_np2 2 84224 3 4000 "-e GGML_CUDA_OCCAMY_DEBUG=1"
arm n2_np2_rep 2 84224 2 80000 ""
arm off_np1_84224 1 84224 0 80000 ""
arm n1_np1_84224 1 84224 1 80000 ""
arm n2_np1_84224 1 84224 2 80000 ""
arm n3_np1_84224 1 84224 3 80000 ""
arm n2_np1_98304 1 98304 2 80000 ""
arm off_np1_98304 1 98304 0 80000 ""
S "all done"
