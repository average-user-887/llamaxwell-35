#!/bin/bash
# Card M1: MTP alloc ladder np2 + 300-token dual-slot draft stats. Env at launch: N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; E=$M/occamy35-48h-20261115/M1
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; echo $OUT > $E/LATEST
S() { echo "$(date +%T) $*" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^nm1-' | xargs -r docker rm -f >/dev/null 2>&1; pkill -f "nvidia-smi.*nm1mon" 2>/dev/null; S "cleanup done; lock released"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 14400 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for j in $(seq 1 120); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; give up"; exit 4; }
ENVS=$(grep -o -- '-e GGML_CUDA_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | tr '\n' ' ')
IMG=sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
# real text
cat $M/occamy35-48h-20261115/PLAN_48H.md $M/occamy35-night-20261113/HANDOFF/ORCHESTRATOR_HANDBOOK.md > $OUT/text.txt
python3 -I - $OUT <<'PY'
import sys,json
o=sys.argv[1]; t=open(o+'/text.txt').read()
json.dump({'a':'Read the following lab planning notes and then write a long, detailed explanation of the plan, step by step.\n\n'+t[:14000],
           'b':'Read the following lab handbook and then write a long, detailed list of the rules and the reasons for each.\n\n'+t[14000:28000],
           'big':(t+'\n')*60}, open(o+'/prompts.json','w'))
PY
run_one() { # tag perslot kvdraft nmax
  tag=$1; ps=$2; kv=$3; nm=$4; c=$((ps*2)); n=nm1-$tag; D=$OUT/$tag; mkdir -p $D
  KVF=""; [ "$kv" = q8 ] && KVF="--spec-draft-type-k q8_0 --spec-draft-type-v q8_0"
  S "$tag start per_slot=$ps c=$c kvdraft=$kv nmax=$nm"
  ( while :; do nvidia-smi -i "$N3_GPU" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $D/vram.csv 2>/dev/null &   # nm1mon
  MON=$!
  docker run -d --name $n --device nvidia.com/gpu=$N3_GPU -p 127.0.0.1:11476:8080 \
    -v <MEDIA>/Storage/uroboros-models:/models:ro \
    -v $M/prod-libs/batch8-20261115:/candidate:ro --entrypoint /candidate/llama-server -e LD_LIBRARY_PATH=/candidate $ENVS $IMG \
    -m /models/occamy-1.0-Q4_0-headq40b-mtp.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c $c -b 2048 -ub 2048 --mixed-batch-size 1024 \
    --prefill-single-stream --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --jinja --metrics \
    --spec-type draft-mtp --spec-draft-n-max $nm $KVF -lv 4 > $D/cid 2>$D/docker.err
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://127.0.0.1:11476/health >/dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' $n 2>/dev/null)" = true ] || break; sleep 2
  done
  docker logs $n > $D/server.log 2>&1
  alloc=$(grep -ci -E "out of memory|cudaMalloc|failed to allocate|CUDA error|failed to create|failed to load" $D/server.log)
  cf=NA; gen=NA
  if [ $ok = 1 ]; then
    python3 -I - $OUT $D <<'PY'
import json,sys,urllib.request as U
o,d=sys.argv[1:3]; P=json.load(open(o+'/prompts.json'))
def post(body,t=1800):
    r=U.Request('http://127.0.0.1:11476/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
    try:
        with U.urlopen(r,timeout=t) as f: return f.status,f.read().decode()
    except Exception as e:
        return getattr(e,'code',0),(e.read().decode() if hasattr(e,'read') else str(e))
s,b=post({'prompt':P['big'],'n_predict':8,'id_slot':0,'cache_prompt':False})
h=U.urlopen('http://127.0.0.1:11476/health',timeout=10).status
s1,b1=post({'prompt':'Hello','n_predict':8,'id_slot':1},300)
json.dump({'status':s,'body':b[:300],'health_after':h,'probe_slot1':s1},open(d+'/ctxfull.json','w'))
PY
    cf=$(python3 -I -c "import json;j=json.load(open('$D/ctxfull.json'));print('status=%s health=%s probe1=%s'%(j['status'],j['health_after'],j['probe_slot1']))" 2>/dev/null || echo err)
  fi
  echo "$tag ok=$ok alloc=$alloc ctxfull=[$cf]" >> $OUT/ladder.txt
  S "$tag ok=$ok alloc=$alloc ctxfull=[$cf]"
  if [ $ok = 1 ] && [ $alloc = 0 ] && echo "$cf" | grep -q "health=200 probe1=200" && [ ! -f $OUT/GENDONE ]; then
    touch $OUT/GENDONE; S "$tag gen test"
    python3 -I - $OUT $D <<'PY'
import json,sys,threading,urllib.request as U
o,d=sys.argv[1:3]; P=json.load(open(o+'/prompts.json')); R={}
def go(slot,k):
    body={'prompt':P[k],'n_predict':300,'id_slot':slot,'temperature':0,'ignore_eos':True,'cache_prompt':False}
    r=U.Request('http://127.0.0.1:11476/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
    with U.urlopen(r,timeout=1800) as f: R[slot]=json.loads(f.read())
t=[threading.Thread(target=go,args=(0,'a')),threading.Thread(target=go,args=(1,'b'))]
[x.start() for x in t]; [x.join() for x in t]
json.dump(R,open(d+'/gen.json','w'))
for s in (0,1):
    r=R.get(s,{}); tm=r.get('timings',{})
    print('slot',s,'tokens',r.get('tokens_predicted'),'draft_n',tm.get('draft_n'),'draft_n_accepted',tm.get('draft_n_accepted'),'pred_per_s',tm.get('predicted_per_second'))
PY
    echo "$tag" > $OUT/GENTAG
    curl -s -m10 http://127.0.0.1:11476/metrics > $D/metrics.txt; curl -s -m10 http://127.0.0.1:11476/slots > $D/slots.json
    docker logs $n > $D/server.log 2>&1
  fi
  kill $MON 2>/dev/null; docker rm -f $n >/dev/null 2>&1; wait $MON 2>/dev/null
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
  [ $ok = 1 ] && [ $alloc = 0 ] && echo "$cf" | grep -q "health=200 probe1=200"
}
FIRSTFAIL=""
for k in 98304 96000 91904 88064 84224 80128; do
  if run_one p$k $k f16 2; then S "ladder pass at $k"; break; else FIRSTFAIL=${FIRSTFAIL:-$k}; fi
done
if [ -n "$FIRSTFAIL" ]; then
  run_one p${FIRSTFAIL}q8 $FIRSTFAIL q8 2
  run_one p${FIRSTFAIL}n1 $FIRSTFAIL f16 1
fi
S done
