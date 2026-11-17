#!/bin/bash
# MV3: env/flag-only arms + correctness dump on the instrumented build. stage=c1
# Env: N3_GPU N3_UNLOAD_URL N3_PORT. One GPU owner: holds gpu-bench.lock for the whole chain.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}" "${N3_PORT:?}"; STAGE=${1:?c1}
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *unload/qwen36-35b-moe) ;; *) exit 2;; esac
M=<WORKDIR>; E=$M/mtp-verifier-20261117; H=$E/harness
OUT=$E/run-$STAGE-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST_$STAGE
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nmv3-' | xargs -r docker stop -t 30 >/dev/null 2>&1; docker ps -a --format '{{.Names}}' | grep '^nmv3-' | xargs -r docker rm >/dev/null 2>&1; jobs -p | xargs -r kill 2>/dev/null; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
LIBS=${N3_LIBS:-<LIBS_DIR>}; HP=127.0.0.1:$N3_PORT; FAIL=0
sha256sum $LIBS/libggml-cuda.so $LIBS/libllama-server-impl.so $LIBS/llama-server > $OUT/build.sha256
nvidia-smi -i $N3_GPU --query-gpu=driver_version,clocks.max.sm,power.limit --format=csv > $OUT/gpu.txt
# arm TAG KIND NP PN NPRED REPS CTXPER EXTRA_DOCKER_ARGS REF_DIR DRAFT_KV EXTRA_SERVER_FLAGS
arm() {
  tag=$1; kind=$2; np=$3; pn=$4; npred=$5; reps=$6; ctx=$7; xe=${8:-}; refd=${9:-}; dkv=${10:-q8_0}; sf=${11:-}
  [ $FAIL = 1 ] && { S "$tag skipped"; return; }
  n=nmv3-$tag; D=$OUT/$tag; mkdir -p $D; chmod 777 $D
  S "$tag start kind=$kind np=$np dkv=$dkv"
  FL="-ub 1024 --cache-ram 12288 --ctx-checkpoints 2 $sf"
  [ $kind = B ] && FL="$FL --spec-type draft-mtp --spec-draft-n-max ${NMAX:-1} --spec-draft-type-k $dkv --spec-draft-type-v $dkv"
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits; sleep 5; done ) > $D/vram.csv 2>/dev/null &
  MON=$!
  echo "kind=$kind np=$np pn=$pn npred=$npred reps=$reps ctx=$ctx dkv=$dkv xe=[$xe] sf=[$sf]" > $D/arm.txt
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p 127.0.0.1:$N3_PORT:8080 \
    -v <MODELS_DIR>:/models:ro -v $LIBS:/candidate:ro -v $D:/mv1out \
    --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $xe $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np $np -c $((ctx*np)) -b 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics $FL -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://127.0.0.1:$N3_PORT/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  if [ $ok = 1 ]; then
    if [ -n "${CUSTOM:-}" ]; then python3 -I $H/mv2_forced.py $HP $D/result.json $refd > $D/client.out 2>&1; else python3 -I $H/mv2_client.py $HP $tag $np $pn $npred $reps $D/result.json $refd > $D/client.out 2>&1; fi
    curl -s -m10 http://127.0.0.1:$N3_PORT/metrics > $D/metrics.txt
  fi
  docker stop -t 60 $n >/dev/null 2>&1
  docker logs $n > $D/server.log 2>&1; docker rm $n >/dev/null 2>&1
  grep -a "mv1_host:" $D/server.log | tail -20 > $D/mv1_host.txt
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  S "$tag ok=$ok peakMiB=$(cut -d, -f1 $D/vram.csv | sort -n | tail -1)"
  if [ $ok != 1 ] || grep -a -q -i "out of memory\|failed to allocate\|CUDA error\|cudaMalloc failed" $D/server.log $D/client.out 2>/dev/null || ! grep -q '"done": true' $D/result.json 2>/dev/null; then
    FAIL=1; S "$tag FAILURE; stopping chain, evidence kept in $D"
  fi
}
C1="-e GGML_CUDA_MV1=1 -e GGML_CUDA_MV1_OUT=/mv1out/counters.tsv"
DMP="-e LLAMA_MV2_DUMP=/mv1out/dump.txt -e LLAMA_MV2_DUMP_LO=0 -e LLAMA_MV2_DUMP_HI=64"
if [ $STAGE = c1 ]; then
  # pre-benchmark kernel validation vs CPU reference, stock vs 2call path (no server)
  for v in stock 2call; do
    X=""; [ $v = 2call ] && X="-e GGML_CUDA_OCCAMY_FA_2CALL=1"
    timeout 1500 docker run --rm --name nmv3-tbo-$v --device nvidia.com/gpu=$N3_GPU --entrypoint /candidate/test-backend-ops -v $LIBS:/candidate:ro -e LD_LIBRARY_PATH=/candidate $ENVS $X $IMG -o FLASH_ATTN_EXT -b CUDA0 > $OUT/tbo-$v.log 2>&1
    S "tbo $v rc=$? $(grep -a -c 'OK' $OUT/tbo-$v.log) ok / $(grep -a -c 'FAIL' $OUT/tbo-$v.log) fail"
  done
  arm A0 A 1 80000 384 3 84224 "$C1"
  arm B0 B 1 80000 384 3 84224 "$C1 $DMP -e LLAMA_MV1_TRACE=1"
  CUSTOM=1 arm dvA A 1 80000 1 0 84224 "$DMP" $OUT/B0/result.json
  arm B_kvq8 B 1 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1 -e GGML_CUDA_OCCAMY_KVQ8=1"
  arm B_2call B 1 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1 -e GGML_CUDA_OCCAMY_FA_2CALL=1"
  arm B_f16 B 1 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1" "" f16
  arm B_all B 1 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1 -e GGML_CUDA_OCCAMY_KVQ8=1 -e GGML_CUDA_OCCAMY_FA_2CALL=1" "" f16
  arm A1 A 1 80000 384 3 84224 "$C1"
fi
if [ $STAGE = c2 ]; then
  arm np2_A A 2 80000 384 3 84224 "$C1"
  arm np2_Bq8 B 2 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1"
  arm np2_Bf16 B 2 80000 384 3 84224 "$C1 -e LLAMA_MV1_TRACE=1" "" f16
  arm s_A A 1 2000 192 3 84224 "$C1"
  arm s_Bq8 B 1 2000 192 3 84224 "$C1 -e LLAMA_MV1_TRACE=1"
  arm s_Bf16 B 1 2000 192 3 84224 "$C1 -e LLAMA_MV1_TRACE=1" "" f16
fi
if [ $STAGE = c3 ]; then
  NT="-e GGML_CUDA_NODE_TIMING=/mv1out/nt.tsv -e GGML_CUDA_NODE_TIMING_SPLIT=1"
  arm nt_A A 1 2000 192 1 84224 "$NT"
  arm nt_B B 1 2000 192 1 84224 "$NT" "" f16
  arm nt_A80 A 1 80000 192 1 84224 "$NT"
  arm nt_B80 B 1 80000 192 1 84224 "$NT" "" f16
fi
if [ $STAGE = c5 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  arm p_A80 A 1 80000 256 2 84224 "$PH"
  arm p_B80 B 1 80000 256 2 84224 "$PH" "" f16
  arm p_A2k A 1 2000 256 2 84224 "$PH"
  arm p_B2k B 1 2000 256 2 84224 "$PH" "" f16
fi
if [ $STAGE = c6 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  NT="-e GGML_CUDA_NODE_TIMING=/mv1out/nt.tsv -e GGML_CUDA_NODE_TIMING_SPLIT=1"
  arm q_B2k_nt B 1 2000 192 1 84224 "$PH $NT" "" f16
  arm q_B2k_nofuse B 1 2000 192 2 84224 "$PH -e GGML_CUDA_DISABLE_FUSION=1" "" f16
  arm q_B80_nt B 1 80000 192 1 84224 "$PH $NT" "" f16
fi
if [ $STAGE = c7 ]; then
  T2="-e GGML_CUDA_OCCAMY_GDN_T2=1"
  timeout 1500 docker run --rm --name nmv3-tbo-t2 --device nvidia.com/gpu=$N3_GPU --entrypoint /candidate/test-backend-ops -v $LIBS:/candidate:ro -e LD_LIBRARY_PATH=/candidate $ENVS $T2 -e GGML_CUDA_OCCAMY_DEBUG=1 $IMG -o OCCAMY_GDN_T2 -b CUDA0 > $OUT/tbo-t2.log 2>&1
  S "tbo t2 rc=$? ok=$(grep -a -c 'OK' $OUT/tbo-t2.log) fail=$(grep -a -c 'FAIL' $OUT/tbo-t2.log) matched=$(grep -a -c 'matched GDN2' $OUT/tbo-t2.log)"
  if [ $(grep -a -c 'FAIL' $OUT/tbo-t2.log) = 0 ] && [ $(grep -a -c 'matched GDN2' $OUT/tbo-t2.log) -gt 0 ]; then
    PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
    arm t2_B80 B 1 80000 256 2 84224 "$PH $T2" "" f16
    arm t2_B2k B 1 2000 256 2 84224 "$PH $T2" "" f16
  else S "t2 test failed or never matched; arms skipped"; fi
fi
if [ $STAGE = c8 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  T2="-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=1"
  arm m2_B80 B 1 80000 256 2 84224 "$PH $T2" "" f16
  arm m2_B2k B 1 2000 256 2 84224 "$PH $T2" "" f16
fi
if [ $STAGE = c9 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  T2="-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=1 -e LLAMA_MTP_MERGE_REEVAL=1"
  arm g_B80 B 1 80000 256 2 84224 "$PH $T2" "" f16
  arm g_B2k B 1 2000 256 2 84224 "$PH $T2" "" f16
fi
if [ $STAGE = c10 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  BE="-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=1 -e LLAMA_MTP_MERGE_REEVAL=1"
  NT="-e GGML_CUDA_NODE_TIMING=/mv1out/nt.tsv -e GGML_CUDA_NODE_TIMING_SPLIT=1"
  arm h_nt80 B 1 80000 128 1 84224 "$PH $BE $NT" "" f16
  NMAX=2 arm h_n2_80 B 1 80000 256 2 84224 "$PH $BE" "" f16
  NMAX=3 arm h_n3_80 B 1 80000 256 2 84224 "$PH $BE" "" f16
fi
if [ $STAGE = c11 ]; then
  PH="-e LLAMA_MV3_PHASE=/mv1out/phase.tsv -e LLAMA_MV1_TRACE=1"
  BE="-e GGML_CUDA_OCCAMY_GDN_T2=1 -e GGML_CUDA_OCCAMY_FA_MROW=3 -e GGML_CUDA_OCCAMY_FA2_NU=4 -e LLAMA_MTP_MERGE_REEVAL=1"
  arm f_B80 B 1 80000 256 2 84224 "$PH $BE" "" f16
  arm f_B2k B 1 2000 256 2 84224 "$PH $BE" "" f16
fi
S "all done"
