#!/usr/bin/env bash
# Card 28d row 2: two-slot decode at ~90K, unprofiled then nvprof, batch8 libs, prod env/args. Env (launch cmd only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_FA_MAXWELL_GEMM
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; N2=$M/occamy35-night-20261114; BIN=$N2/9/bin; SL=$M/occamy35-day-20261112/evidence/G1b/slots
OUT=$N2/28d/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; chmod 777 $OUT; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
CN=n28d-dec
cleanup() { docker logs $CN > $OUT/server-last.log 2>&1; docker rm -f $CN >/dev/null 2>&1; S "cleanup; release lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 5400 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 300); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; abort"; exit 4; }
ENV=""; for x in $(docker inspect llamacpp-m6000 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^GGML_' | grep -v MAXWELL_GRAPHS); do ENV="$ENV -e $x"; done
echo "$ENV" > $OUT/env.txt
for ARM in ${ARMS:-plain prof}; do
  docker rm -f $CN >/dev/null 2>&1
  if [ $ARM = prof ]; then EP=/usr/local/cuda-12.8/bin/nvprof; PRE="--print-gpu-trace --csv --log-file /out/trace.csv /candidate/llama-server"; else EP=/candidate/llama-server; PRE=""; fi
  docker run -d --init --name $CN -p 127.0.0.1:18494:8080 --device nvidia.com/gpu=$N3_GPU -v <MODELS>/uroboros-models:/models:ro -v $SL:/slots:ro \
    -v $BIN:/candidate:ro -v $OUT:/out -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib $ENV \
    --entrypoint $EP nvidia/cuda:12.8.1-devel-ubuntu24.04 $PRE \
    -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 2048 \
    --mixed-batch-size 1024 --prefill-single-stream --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 0.0.0.0 --port 8080 --slot-save-path /slots -lv 4 > $OUT/run-$ARM.id 2>&1 || { S "run failed $ARM"; exit 5; }
  S "started $ARM"
  python3 -I - $OUT $ARM <<'PY'
import sys, json, time, http.client, threading, subprocess
out, arm = sys.argv[1], sys.argv[2]
def call(path, body, timeout=1800):
    c = http.client.HTTPConnection('127.0.0.1', 18494, timeout=timeout); c.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'}); r = c.getresponse(); d = r.read(); c.close(); return r.status, d
for _ in range(400):
    try:
        c = http.client.HTTPConnection('127.0.0.1', 18494, timeout=5); c.request('GET', '/health'); 
        if c.getresponse().status == 200: break
    except Exception: pass
    time.sleep(3)
time.sleep(3)
tok = json.load(open('<LAB>/codex-investigation-20261112/fixture_tokens.json'))
for sl in (0, 1):
    st, d = call('/slots/%d?action=restore' % sl, {'filename': 's1_90k_off200000.bin'}); print('restore', sl, st, d[:80])
res = {}
def go(sl, tail):
    p = tok[200000:290000] + [tail]; ts = []
    c = http.client.HTTPConnection('127.0.0.1', 18494, timeout=1800)
    c.request('POST', '/completion', json.dumps({'prompt': p, 'n_predict': 256, 'id_slot': sl, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True, 'stream': True}), {'Content-Type': 'application/json'})
    r = c.getresponse(); fin = None
    for line in r:
        if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
        j = json.loads(line[5:])
        if j.get('stop'): fin = j
        else: ts.append(time.monotonic())
    tm = (fin or {}).get('timings', {})
    res[sl] = {'events': len(ts), 'rate_events_20_to_end': (len(ts) - 21) / (ts[-1] - ts[20]) if len(ts) > 25 else None, 'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'),
               'predicted_n': tm.get('predicted_n'), 'predicted_per_second': tm.get('predicted_per_second'), 'tokens_cached': (fin or {}).get('tokens_cached')}
th = [threading.Thread(target=go, args=(0, 198)), threading.Thread(target=go, args=(1, 220))]
[t.start() for t in th]; [t.join() for t in th]
json.dump(res, open('%s/decode-%s.json' % (out, arm), 'w'), indent=1); print(res)
subprocess.run(['docker', 'exec', 'n28d-dec', 'bash', '-c', 'kill -INT $(pidof llama-server)'], capture_output=True)
for _ in range(120):
    if not subprocess.run(['docker', 'ps', '-q', '-f', 'name=n28d-dec'], capture_output=True, text=True).stdout.strip(): break
    time.sleep(2)
PY
  S "done $ARM rc=$?"
  docker logs $CN > $OUT/server-$ARM.log 2>&1
done
