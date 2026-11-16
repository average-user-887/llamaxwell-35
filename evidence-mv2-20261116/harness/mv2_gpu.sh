#!/bin/bash
# MV2: counters-verified diagnosis of 35B MTP vs no-spec on the M6000. stage=smoke|long
# Env (launch only): N3_GPU, N3_UNLOAD_URL (model-unload POST only), N3_PORT, N3_HOST
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}" "${N3_HOST:?}"; STAGE=${1:?smoke|long}
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *unload/qwen36-35b-moe) ;; *) exit 2;; esac
M=<PATH> E=$M/occamy35-48h-20261115/MV2
OUT=$E/run-$STAGE-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST_$STAGE
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nmv2-' | xargs -r docker stop -t 30 >/dev/null 2>&1; docker ps -a --format '{{.Names}}' | grep '^nmv2-' | xargs -r docker rm >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
LIBS=$E/build/bin; HP=$N3_HOST:$N3_PORT; FAIL=0
# arm TAG KIND NP PN NPRED REPS CTXPER EXTRA_DOCKER_ARGS [REF_DIR]
arm() {
  tag=$1; kind=$2; np=$3; pn=$4; npred=$5; reps=$6; ctx=$7; xe=${8:-}; refd=${9:-}
  [ $FAIL = 1 ] && { S "$tag skipped"; return; }
  n=nmv2-$tag; D=$OUT/$tag; mkdir -p $D; chmod 777 $D
  S "$tag start kind=$kind np=$np"
  FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2"
  [ $kind = B ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits; sleep 5; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p <HOST_IP>:$N3_PORT:8080 \
    -v <PATH> -v $LIBS:/candidate:ro -v $D:/mv1out \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $xe $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np $np -c $((ctx*np)) -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host <IP> --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://<HOST_IP>:$N3_PORT/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    if [ -n "${CUSTOM:-}" ]; then python3 -I $E/mv2_forced.py $HP $D/result.json $refd > $D/client.out 2>&1; else python3 -I $E/mv2_client.py $HP $tag $np $pn $npred $reps $D/result.json $refd > $D/client.out 2>&1; fi
    curl -s -m10 http://<HOST_IP>:$N3_PORT/metrics > $D/metrics.txt
  fi
  docker stop -t 60 $n >/dev/null 2>&1
  docker logs $n > $D/server.log 2>&1; docker rm $n >/dev/null 2>&1
  grep -a "mv1_host:" $D/server.log | tail -20 > $D/mv1_host.txt
  grep -a -i "occamy_\|fallback\|spec\|rs_seq" $D/server.log | cut -c1-240 | sort | uniq -c | sort -rn | head -40 > $D/counters.txt
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok cnt_lines=$(wc -l < $D/counters.tsv 2>/dev/null)"
  if [ $ok != 1 ] || grep -a -q -i "out of memory\|failed to allocate\|CUDA error\|cudaMalloc failed" $D/server.log $D/client.out 2>/dev/null || ! grep -q '"done": true' $D/result.json 2>/dev/null; then
    FAIL=1; S "$tag FAILURE; stopping chain, evidence kept in $D"
  fi
}
C1="-e GGML_CUDA_MV1=1 -e GGML_CUDA_MV1_OUT=/mv1out/counters.tsv"
if [ $STAGE = smoke ]; then
  arm smB B 1 2000 192 0 84224 "$C1 -e LLAMA_MV1_TRACE=1"
  arm smA A 1 2000 192 0 84224 "$C1" $OUT/smB
fi
if [ $STAGE = long ]; then
  CT="-e GGML_CUDA_MV1=2 -e GGML_CUDA_MV1_OUT=/mv1out/counters.tsv -e LLAMA_MV1_TRACE=1 -e GGML_CUDA_NODE_TIMING=/mv1out/nt.tsv -e GGML_CUDA_NODE_TIMING_SPLIT=1"
  arm shB B 1 2000 192 0 84224 "$C1 -e LLAMA_MV1_TRACE=1"
  arm shA A 1 2000 192 0 84224 "$C1" $OUT/shB
  for k in A B B A; do arm np1_${k}0_$RANDOM $k 1 80000 384 4 84224 "-e LLAMA_MV1_TRACE=1"; done
  arm np1_B1 B 1 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1"
  arm np1_A1 A 1 80000 384 3 84224 "$C1"
  arm np1_trB B 1 80000 256 1 84224 "$CT"
  for k in A B; do arm np2_${k}0 $k 2 80000 384 2 84224 "-e LLAMA_MV1_TRACE=1"; done
  arm np2_B1 B 2 80000 384 1 84224 "$C1 -e LLAMA_MV1_TRACE=1"
fi
if [ $STAGE = div ]; then
  REFB=$(ls -d $E/run-long-*/np1_B0_15642 | tail -1)/result.json
  CUSTOM=1 arm dvA A 1 80000 1 0 84224 "" $REFB
fi
S "all done"
