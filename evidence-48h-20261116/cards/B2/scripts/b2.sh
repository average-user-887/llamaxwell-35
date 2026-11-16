#!/bin/bash
# Card B2: per-component decode step timing (LD_PRELOAD launch profiler, no nsys/CUPTI in image) at k=1/k=2 verify rows + accept rate on repaired fixture.
# Env required: N3_GPU (uuid), N3_UNLOAD_URL (.../api/models/unload/qwen36-35b-moe). Containers nb2-. Lock: flock -w 43200 on gpu-bench.lock.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *unload/qwen36-35b-moe) ;; *) exit 2;; esac
M=<M6000_ROOT>; W=$M/occamy35-48h-20261115; E=$W/B2
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nb2-' | xargs -r docker rm -f >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done"; exec 9>&-; }
trap cleanup EXIT
S "waiting for lock"
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"; T0=$(date +%s)
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
Q="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
PORT=11486
arm() { # tag np perslot spec depths shim(1|0) extra_env fixture(0|1)
  tag=$1; np=$2; ps=$3; sp=$4; dp=$5; shim=$6; xe=$7; fx=$8; n=nb2-$tag; D=$OUT/$tag; mkdir -p $D; chmod 777 $D
  if [ $(( $(date +%s) - T0 )) -gt 5400 ]; then S "$tag skipped (time budget)"; return; fi
  S "$tag start np=$np per_slot=$ps spec=$sp depths=$dp shim=$shim"
  FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2"
  [ "$sp" != 0 ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max $sp $Q"
  PRE=""; [ "$shim" = 1 ] && PRE="-v $E:/kp:ro -v $D:/kpo -e LD_PRELOAD=/kp/kprof.so -e KPROF_OUT=/kpo/prof.txt -e KPROF_PHASE=/kpo/phase"
  echo idle > $D/phase; chmod 666 $D/phase
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p 127.0.0.1:$PORT:8080 \
    -v <MEDIA>/Storage/uroboros-models:/models:ro -v $M/prod-libs/batch8-20261115:/candidate:ro $PRE \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $xe $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np $np -c $((ps*np)) -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    python3 -I $E/b2_client.py $PORT $np $dp $D/result.json $D/phase > $D/client.out 2>&1
    curl -s -m10 http://127.0.0.1:$PORT/metrics > $D/metrics.txt
    if [ "$fx" = 1 ]; then
      S "$tag fixture start"
      N3R_LOCK_HELD=1 N3R_OUT=$D/fixture $W/B0/run_n3r.sh http://127.0.0.1:$PORT $tag --ctx 36864 --turns 3 --server-log $D/server_live.log > $D/fixture.out 2>&1
      S "$tag fixture done rc=$?"
    fi
  else S "$tag server failed to start"; fi
  docker logs $n > $D/server.log 2>&1
  grep -a -i "occamy_\|fallback\|draft\|spec" $D/server.log | grep -a -v "^srv.*params" | cut -c1-240 | sort | uniq -c | sort -rn | head -60 > $D/counters.txt
  kill $MON 2>/dev/null; docker rm -f $n >/dev/null 2>&1; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok profbytes=$(stat -c %s $D/prof.txt 2>/dev/null) peak=$(sort -n $D/vram.csv | tail -1)"
}
# (the fixture phase reads server log for occamy_spec lines only; accepted/drafted come from per-request timings)
arm n2_np1 1 98304 2 80000,90000 1 "" 0
arm n1_np1 1 98304 1 80000,90000 1 "" 0
arm n2_np2 2 84224 2 80000 1 "" 0
arm n1_np2 2 84224 1 80000 1 "" 0
arm off_np1 1 98304 0 80000,90000 1 "" 0
arm off_np2 2 84224 0 80000 1 "" 0
S "all done"
