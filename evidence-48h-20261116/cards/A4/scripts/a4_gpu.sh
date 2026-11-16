#!/usr/bin/env bash
# A4 MoE FP32 prefill GPU checks (NOT run by the coder; the orchestrator queues it). ~35 min GPU hold.
# (1) test-backend-ops -o MUL_MAT_ID, A4 libs, flag on with MIN=1 (every eligible case through the FP32 path); engaged lines must be >0.
# (2) nvprof in-model profile, 16384-token single-slot prefill (8 ubatches of 2048), prod batch8 env, A4 libs, flag off vs on.
#     Metric: MoE ms per ubatch (an28 classes moe_mmq+moe, median of ubatches 1..6). Gate: off/on >= 1.07x.
# (3) N3 short replay ABBA x1, both arms = A4 libs + bundle9 env; B adds GGML_CUDA_MOE_FP32_PREFILL=1,
#     first B also GGML_CUDA_MOE_FP32_CHECK=64 (relRMSE FP32 vs mmq per op; expected ~5e-3 = the mmq q8_1 error, nonfinite 0).
#     Void-arm rule: a B arm without a 'moe-fp32: engaged' line is VOID, not a result.
# Model dmean/KLD gate is a separate run (not in this script).
# Env (launch only): N3_GPU, N3_UNLOAD_URL. POST unload only. Does not restore prod.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_MOE_FP32_PREFILL GGML_CUDA_MOE_FP32_PREFILL_MIN GGML_CUDA_MOE_FP32_CHECK GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; E=$M/occamy35-48h-20261115/A4; D=$N2/28h/h
LIBS=$E/build/bin; CUDAIMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; MODELS=<MEDIA>/Storage/uroboros-models
FIX=$M/occamy35-day-20261112/evidence/fixtures/tokens/d0.json
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/tbo $OUT/prof; chmod 777 $OUT/prof; echo $OUT > $E/LATEST; S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
cleanup() { docker ps -a --format '{{.Names}}' | grep '^na4-' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }
trap cleanup EXIT
exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
S "lock taken"
curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
for i in $(seq 1 20); do
  for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
  S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
done
[ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }

# (1) test-backend-ops MUL_MAT_ID through the FP32 path
timeout 1200 docker run --rm --name "na4-tbo-$$" --network none --device "nvidia.com/gpu=$N3_GPU" -v $LIBS:/candidate:ro \
  -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib -e GGML_CUDA_MOE_FP32_PREFILL=1 -e GGML_CUDA_MOE_FP32_PREFILL_MIN=1 \
  -e GGML_CUDA_OCCAMY_DEBUG=1 --entrypoint /candidate/test-backend-ops $CUDAIMG -o MUL_MAT_ID -b CUDA0 > $OUT/tbo/tbo.log 2>&1
S "tbo rc=$? engaged=$(grep -c 'moe-fp32: engaged' $OUT/tbo/tbo.log) fail=$(grep -c 'FAIL' $OUT/tbo/tbo.log) $(grep -h 'tests passed' $OUT/tbo/tbo.log | tail -1)"

# (2) nvprof in-model profile, off then on
PENV=$(grep -o '\-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | grep -v MAXWELL_GRAPHS | sort -u | tr '\n' ' ')
for tag in off on; do
  P=$OUT/prof/$tag; mkdir -p $P; chmod 777 $P; CN=na4-prof-$tag
  X="$PENV -e GGML_CUDA_OCCAMY_DEBUG=1"; [ $tag = on ] && X="$X -e GGML_CUDA_MOE_FP32_PREFILL=1"
  echo "$X" > $P/env.txt
  docker run -d --init --name $CN --network none --device nvidia.com/gpu=$N3_GPU -v $MODELS:/models:ro -v $LIBS:/candidate:ro -v $P:/out \
    -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib $X \
    --entrypoint /usr/local/cuda-12.8/bin/nvprof $CUDAIMG --print-gpu-trace --csv --log-file /out/trace.csv \
    /candidate/llama-server -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 2048 \
    --mixed-batch-size 1024 --prefill-single-stream --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 127.0.0.1 --port 8080 > $P/run.id 2>&1 || { S "prof $tag start failed"; continue; }
  timeout 1500 python3 -I - $CN $P $FIX <<'PY'
import sys, json, subprocess, time
cn, out, fix = sys.argv[1], sys.argv[2], sys.argv[3]
ids = json.load(open(fix))[:16384]
json.dump({'prompt': ids, 'n_predict': 1, 'id_slot': 0, 'cache_prompt': False, 'temperature': 0}, open(out + '/req.json', 'w'))
def ex(c, t=None): return subprocess.run(['docker', 'exec', cn, 'bash', '-c', c], capture_output=True, text=True, timeout=t)
for _ in range(300):
    if ' 200 ' in ex("exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && head -1 <&3", 10).stdout: break
    time.sleep(2)
time.sleep(5); t0 = time.time()
r = ex("L=$(stat -c %s /out/req.json); exec 3<>/dev/tcp/127.0.0.1/8080; printf 'POST /completion HTTP/1.0\\r\\nContent-Type: application/json\\r\\nContent-Length: %s\\r\\n\\r\\n' $L >&3; cat /out/req.json >&3; cat <&3", 1200)
open(out + '/resp.txt', 'w').write(r.stdout[-4000:] + r.stderr[-1000:]); print('wall', time.time() - t0)
ex('kill -INT $(pidof llama-server)')
for _ in range(120):
    if not subprocess.run(['docker', 'ps', '-q', '-f', 'name=' + cn], capture_output=True, text=True).stdout.strip(): break
    time.sleep(2)
PY
  docker logs $CN > $P/server.log 2>&1; docker rm -f $CN >/dev/null 2>&1
  python3 -I $N2/28/an28.py $P/trace.csv > $P/an28.txt 2>&1
  S "prof $tag engaged=$(grep -c 'moe-fp32: engaged' $P/server.log) $(grep -h 'moe-fp32: engaged' $P/server.log | tail -1)"
done
python3 -I - $OUT/prof > $OUT/prof/moe_gate.txt 2>&1 <<'PY'
import sys, json, statistics as st
o = sys.argv[1]; r = {}
for t in ('off', 'on'):
    segs = json.load(open(f'{o}/{t}/trace.csv.segs.json'))
    mid = segs[1:-1] if len(segs) > 3 else segs
    r[t] = (st.median(s['by'].get('moe_mmq', 0) + s['by'].get('moe', 0) for s in mid), st.median(s['busy'] for s in mid), len(mid))
    print(f"{t}: MoE {r[t][0]:.1f} ms/ubatch, busy {r[t][1]:.1f} ms/ubatch, n={r[t][2]}")
print(f"MoE speedup off/on {r['off'][0]/r['on'][0]:.4f}x (gate >=1.07), busy speedup {r['off'][1]/r['on'][1]:.4f}x")
PY
S "moe gate: $(tail -1 $OUT/prof/moe_gate.txt)"

# (3) N3 short replay ABBA x1
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1 GGML_CUDA_OFA_RAGGED=1 GGML_CUDA_OFA_RAGGED_RATIO=2 GGML_CUDA_OCCAMY_GDN18=1 GGML_CUDA_GDN2_A2ROW=1"
i=0
for a in A B B A; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n
  if [ $a = A ]; then X="$BASE"; elif [ $i = 2 ]; then X="$BASE GGML_CUDA_MOE_FP32_PREFILL=1 GGML_CUDA_MOE_FP32_CHECK=64"; else X="$BASE GGML_CUDA_MOE_FP32_PREFILL=1"; fi
  export N3_EXTRA_ENV="$X" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$?"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  ENG=$(grep -h 'moe-fp32: engaged' $L | wc -l)
  S "$n engaged lines=$ENG last: $(grep -h 'moe-fp32: engaged' $L | tail -1)"
  [ $a = B ] && [ "$ENG" = 0 ] && S "$n VOID (flag not engaged)"
  [ $i = 2 ] && { grep -h 'moe-fp32-check' $L > $OUT/check.log; S "CHECK lines $(wc -l < $OUT/check.log), nonfinite>0: $(grep -vc 'nonfinite 0$' $OUT/check.log), max relRMSE $(awk '{for(i=1;i<=NF;i++) if($i~/relRMSE/) print $(i+3)}' $OUT/check.log | sort -g | tail -1)"; }
done
S done
