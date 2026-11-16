#!/bin/bash
# Card MV1a: why is 35B MTP verification slower than no-spec on this GPU. Paired ABBA no-spec (A) vs MTP draft 1 (B),
# 2 slots x 84224 ctx populated to ~80K each by prefill, plus single-slot pair, sampling arm, then short node-timing trace of B.
# Env (launch only, never in files): N3_GPU, N3_UNLOAD_URL (model-unload POST only), N3_PORT, N3_HOST (client-side host for the published port).
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}" "${N3_HOST:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *unload/qwen36-35b-moe) ;; *) exit 2;; esac
M=$STORAGE/m6000-llama-cpp; E=$M/occamy35-48h-20261115/MV1
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nmv-' | xargs -r docker rm -f >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
PROD=$M/prod-libs/batch8-20261115; MV1LIBS=$E/build/bin
HP=$N3_HOST:$N3_PORT
FAIL=0
# arm TAG KIND(A|B) NP PROMPT_TOKENS NPRED SAMPLING LIBDIR EXTRA_DOCKER_ARGS
arm() {
  tag=$1; kind=$2; np=$3; pn=$4; npred=$5; samp=$6; libs=$7; xe=${8:-}
  [ $FAIL = 1 ] && { S "$tag skipped (earlier failure)"; return; }
  n=nmv-$tag; D=$OUT/$tag; mkdir -p $D; chmod 777 $D
  S "$tag start kind=$kind np=$np"
  FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2"
  [ $kind = B ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p localhost:$N3_PORT:8080 \
    -v $STORAGE/uroboros-models:/models:ro -v $libs:/candidate:ro -v $D:/mv1out \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $xe $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np $np -c $((84224*np)) -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://localhost:$N3_PORT/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    python3 -I $E/mv1_client.py $HP $tag $np $pn $npred $D/result.json $samp > $D/client.out 2>&1
    curl -s -m10 http://localhost:$N3_PORT/metrics > $D/metrics.txt
  fi
  docker logs $n > $D/server.log 2>&1
  grep -a "mv1_host:\|mv1_fa:" $D/server.log | tail -40 > $D/mv1_counters.txt
  grep -a -i "occamy_\|fallback\|spec\|rs_seq" $D/server.log | cut -c1-240 | sort | uniq -c | sort -rn | head -60 > $D/counters.txt
  kill $MON 2>/dev/null; docker rm -f $n >/dev/null 2>&1; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok peak_MiB=$(sort -n $D/vram.csv | tail -1)"
  if [ $ok != 1 ] || grep -a -q -i "out of memory\|failed to allocate\|CUDA error\|cudaMalloc failed" $D/server.log $D/client.out 2>/dev/null || ! grep -q '"done": true' $D/result.json 2>/dev/null || grep -q '"err"' $D/result.json 2>/dev/null; then
    FAIL=1; S "$tag FAILURE (OOM/crash/client error); stopping chain, evidence kept in $D"
  fi
}
# two populated slots, 3 ABBA pairs
for k in A B B A A B; do arm np2_${k}_$RANDOM $k 2 80000 384 0 $PROD; done
# single slot ABBA
for k in A B B A; do arm np1_${k}_$RANDOM $k 1 80000 384 0 $PROD; done
# sampling at server defaults (single slot)
for k in A B; do arm samp_${k} $k 1 80000 384 1 $PROD; done
# short trace of B with node timing + per-path FA counters + host timers (mv1 build; not a throughput run)
mkdir -p $OUT/nt; chmod 777 $OUT/nt
FAIL=0
arm trace_B B 1 80000 256 0 $MV1LIBS "-v $OUT/nt:/n3l -e GGML_CUDA_NODE_TIMING=/n3l/trace_B.nt.tsv -e GGML_CUDA_NODE_TIMING_SPLIT=1 -e GGML_CUDA_MV1=1 -e LLAMA_MV1_TRACE=1"
S "all done"
