#!/bin/bash
# Card B1b: N3R real-generation spec-type matrix on batch8 libs.
set -u
unset GGML_CUDA_MAXWELL_GRAPHS
M=<M6000_ROOT>; W=$M/occamy35-48h-20261115; E=$W/B1b
GPU=<GPU_UUID>
UNL=http://<IP>:11445/api/models/unload/qwen36-35b-moe
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nb1-' | xargs -r docker rm -f >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$UNL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
Q="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
HELP=$(docker run --rm --entrypoint /candidate/llama-server -v $M/prod-libs/batch8-20261115:/candidate:ro -e LD_LIBRARY_PATH=/candidate $IMG --help 2>&1)
echo "$HELP" | grep -A2 -- '--spec-type' > $OUT/spec_type_help.txt
arm() { # tag specflags
  tag=$1; sf=$2; n=nb1-$tag; D=$OUT/$tag; mkdir -p $D; S "$tag start: $sf"
  ( while :; do nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$GPU -p 127.0.0.1:11477:8080 \
    -v <MEDIA>/Storage/uroboros-models:/models:ro -v $M/prod-libs/batch8-20261115:/candidate:ro \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 168448 -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics -ub 1024 --cache-ram 12288 --ctx-checkpoints 2 $sf -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://127.0.0.1:11477/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    docker logs -f $n > $D/server.log 2>&1 &
    LG=$!
    N3R_LOCK_HELD=1 N3R_OUT=$D/run bash $W/B0/run_n3r.sh http://127.0.0.1:11477 $tag --ctx 84224 --server-log $D/server.log > $D/client.out 2>&1
    kill $LG 2>/dev/null
  fi
  docker logs $n > $D/server.log 2>&1
  kill $MON 2>/dev/null; docker rm -f $n >/dev/null 2>&1; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok peak=$(sort -n $D/vram.csv | tail -1)"
}
arm offA ""
NV=$(python3 -I -c "import json;r=json.load(open('$OUT/offA/run/run.json'));print(r.get('n_void',99), len(r['turns_rec']))" 2>/dev/null)
set -- $NV
if [ -z "${1:-}" ] || [ $((${1:-99}*2)) -gt ${2:-12} ]; then S "ABORT: offA void turns ${1:-?}/${2:-?} (>half or no run.json); fixture still unrepresentative"; exit 5; fi
S "offA void ${1}/${2} ok, continuing"
arm offA2 ""
arm mtp2 "--spec-type draft-mtp --spec-draft-n-max 2 $Q"
arm mtp3 "--spec-type draft-mtp --spec-draft-n-max 3 $Q"
arm mtp1 "--spec-type draft-mtp --spec-draft-n-max 1 $Q"
arm ngmod "--spec-type ngram-mod"
arm ngmapk "--spec-type ngram-map-k --spec-draft-n-max 16 --spec-ngram-map-k-size-m 16"
S "ngmod_mtp2 skipped: OOM at ub1024/84224 (from B1)"
S "all done"
python3 -I $E/summarize.py $OUT > $OUT/summary.md 2>$OUT/summarize.err
C=$W/B0/n3r_compare.py; A1=$OUT/offA/run; A2=$OUT/offA2/run
for a in mtp2 mtp3 mtp1 ngmod ngmapk; do echo "=== $a vs floor(offA,offA2)"; python3 -I $C $A1 $A2 $OUT/$a/run; done > $OUT/compare_floor.txt 2>&1
echo "=== mtp1 vs mtp2 (n=1 vs n=2 floor, gate 4a)" > $OUT/compare_mtp1_mtp2.txt; python3 -I $C $OUT/mtp2/run $OUT/mtp1/run >> $OUT/compare_mtp1_mtp2.txt 2>&1
S "summary written"
