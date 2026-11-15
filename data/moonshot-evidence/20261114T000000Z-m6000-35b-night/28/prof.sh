#!/usr/bin/env bash
# Card 28 step 1: nvprof GPU trace of one 94208-token single-slot prefill (46 ubatches of 2048, depth 0..92K) on batch8 libs (9/bin), prod env.
# Env (launch command only): N3_GPU, N3_UNLOAD_URL.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<LAB>; N2=$M/occamy35-night-20261114; E=$N2/28; BIN=${N28_BIN:-$N2/9/bin}; TAG=${N28_TAG:-a}
OUT=$E/prof/$TAG-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT; chmod 777 $OUT; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
CN=n28-prof-$TAG
cleanup() { docker logs $CN > $OUT/server.log 2>&1; docker rm -f $CN >/dev/null 2>&1; S "cleanup; release lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 14400 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 300); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 1; done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy; abort"; exit 4; }
ENV=""; for x in $(docker inspect llamacpp-m6000 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^GGML_' | grep -v MAXWELL_GRAPHS); do ENV="$ENV -e $x"; done
ENV="$ENV ${N28_EXTRA_ENV:-}"
echo "$ENV" > $OUT/env.txt
docker run -d --init --name $CN --network none --device nvidia.com/gpu=$N3_GPU -v <MODELS>/uroboros-models:/models:ro \
  -v $BIN:/candidate:ro -v $OUT:/out -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib $ENV \
  --entrypoint /usr/local/cuda-12.8/bin/nvprof nvidia/cuda:12.8.1-devel-ubuntu24.04 --print-gpu-trace --csv --log-file /out/trace.csv \
  /candidate/llama-server -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 2048 \
  --mixed-batch-size 1024 --prefill-single-stream --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 127.0.0.1 --port 8080 -lv 4 > $OUT/run.id 2>&1 || { S "run failed"; exit 5; }
S "started"
python3 -I - $CN $OUT <<'PY'
import sys, json, subprocess, time
cn, out = sys.argv[1], sys.argv[2]
ids = json.load(open('<LAB>/occamy35-day-20261112/evidence/fixtures/tokens/d0.json'))[:94208]
json.dump({'prompt': ids, 'n_predict': 1, 'id_slot': 0, 'cache_prompt': False, 'temperature': 0}, open(out + '/req.json', 'w'))
def ex(c, t=None): return subprocess.run(['docker', 'exec', cn, 'bash', '-c', c], capture_output=True, text=True, timeout=t)
for _ in range(300):
    if ' 200 ' in ex("exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && head -1 <&3", 10).stdout: break
    time.sleep(2)
time.sleep(5)
t0 = time.time()
# no curl in image: use python? image is ubuntu devel; use bash /dev/tcp HTTP POST
r = ex("L=$(stat -c %s /out/req.json); exec 3<>/dev/tcp/127.0.0.1/8080; printf 'POST /completion HTTP/1.0\\r\\nContent-Type: application/json\\r\\nContent-Length: %s\\r\\n\\r\\n' $L >&3; cat /out/req.json >&3; cat <&3", 1800)
open(out + '/resp.txt', 'w').write(r.stdout[-4000:] + r.stderr[-1000:]); print('wall', time.time() - t0)
ex('kill -INT $(pidof llama-server)')
for _ in range(120):
    if not subprocess.run(['docker', 'ps', '-q', '-f', 'name=' + cn], capture_output=True, text=True).stdout.strip(): break
    time.sleep(2)
PY
S "done rc=$?"
