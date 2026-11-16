# FK1 shared GPU-run helpers (sourced by fk1_gpu.sh and fk1_3way.sh; NOT run by the coder).
# Env (launch only, never written to a file): N3_GPU, N3_UNLOAD_URL. POST unload of the 35B model only. Does not restore prod.
# Power limit and clocks are not touched. GGML_CUDA_MAXWELL_GRAPHS stays absent.
set -u
: "${N3_GPU:?}" "${N3_UNLOAD_URL:?}"
unset GGML_CUDA_MAXWELL_GRAPHS GGML_CUDA_MOE_FABLE_PREFILL GGML_CUDA_MOE_FABLE_PREFILL_MIN GGML_CUDA_MOE_FABLE_CHECK \
      GGML_CUDA_MOE_FP32_PREFILL GGML_CUDA_MOE_FP32_PREFILL_MIN GGML_CUDA_MOE_FP32_CHECK \
      GGML_CUDA_OFA_RAGGED GGML_CUDA_OCCAMY_GDN18 GGML_CUDA_GDN2_A2ROW LLAMA_ARG_MIXED_BATCH_PREFILL_MAX
case "$N3_UNLOAD_URL" in *unload-all*) echo refuse; exit 2;; *qwen36-35b-moe*) ;; *) echo "unload target is not the 35B model"; exit 2;; esac
M=<M6000_ROOT>; N2=$M/occamy35-night-20261114; W=$M/occamy35-48h-20261115; E=$W/FK1; D=$N2/28h/h
LIBS_PROD=$M/prod-libs/batch8-20261115; LIBS_FK1=$E/build/bin; LIBS_A4=$W/A4/build/bin
CUDAIMG=nvidia/cuda:12.8.1-devel-ubuntu24.04; MODELS=<MEDIA>/Storage/uroboros-models
FIX=$M/occamy35-day-20261112/evidence/fixtures/tokens/d0.json
ENG_RE='moe-(fable|fp32): engaged'
export N3_GPU PYTHONDONTWRITEBYTECODE=1 N3_PROFILE=short
PENV=$(grep -o '\-e GGML_[A-Z0-9_]*=[^ ]*' $M/run-35b-batch8-20261115.sh | grep -v MAXWELL_GRAPHS | sort -u | tr '\n' ' ')

S() { echo "$* $(date +%T)" >> $OUT/steps.log; }
fk1_cleanup() { docker ps -a --format '{{.Names}}' | grep '^nfb-' | grep -v '^nfb-build$' | xargs -r docker rm -f >/dev/null 2>&1; S "cleanup done; releasing lock"; exec 9>&-; }

fk1_take_gpu() { # lock, then POST-only unload of the 35B model, wait until the GPU has no compute process
  trap fk1_cleanup EXIT
  exec 9>$M/gpu-bench.lock; flock -w 43200 9 || { S "lock busy"; exit 90; }
  S "lock taken"
  curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
  local i j
  for i in $(seq 1 20); do
    for j in $(seq 1 60); do [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break 2; sleep 1; done
    S "GPU busy; retry $i"; curl -s -m60 -X POST "$N3_UNLOAD_URL" > $OUT/unload.json
  done
  [ "$(nvidia-smi -i "$N3_GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { S "GPU busy after 20 min; give up"; exit 4; }
}

# fk1_tbo <outdir> <libs> <-e flags...>: test-backend-ops -o MUL_MAT_ID on the GPU
fk1_tbo() {
  local o=$1 libs=$2; shift 2; mkdir -p $o
  timeout 1800 docker run --rm --name "nfb-tbo-$$" --network none --device "nvidia.com/gpu=$N3_GPU" -v $libs:/candidate:ro \
    -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib "$@" -e GGML_CUDA_OCCAMY_DEBUG=1 \
    --entrypoint /candidate/test-backend-ops $CUDAIMG -o MUL_MAT_ID -b CUDA0 > $o/tbo.log 2>&1
  S "tbo $(basename $o) rc=$? engaged=$(grep -Ec "$ENG_RE" $o/tbo.log) fail=$(grep -c 'FAIL' $o/tbo.log) $(grep -h 'tests passed' $o/tbo.log | tail -1)"
}

# fk1_server_run <outdir> <libs> <ntokens> <nvprof 0|1> <extra -e flags as one string>
# One llama-server (prod batch8 env + extra), one single-slot prefill of <ntokens> fixture tokens, n_predict 1.
# With nvprof=1 this is the A4 microbench: 16384 tokens = 8 ubatches of 2048, trace -> an28 classes.
fk1_server_run() {
  local P=$1 libs=$2 ntok=$3 prof=$4 extra=$5 CN=nfb-run-$$
  mkdir -p $P; chmod 777 $P
  local X="$PENV -e GGML_CUDA_OCCAMY_DEBUG=1 $extra"; echo "$X" > $P/env.txt; echo "$libs" > $P/libs.txt
  local EP=/candidate/llama-server PRE=""
  [ $prof = 1 ] && { EP=/usr/local/cuda-12.8/bin/nvprof; PRE="--print-gpu-trace --csv --log-file /out/trace.csv /candidate/llama-server"; }
  docker run -d --init --name $CN --network none --device nvidia.com/gpu=$N3_GPU -v $MODELS:/models:ro -v $libs:/candidate:ro -v $P:/out \
    -e LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib $X \
    --entrypoint $EP $CUDAIMG $PRE -m /models/occamy-1.0-Q4_0-headq40b.gguf --alias qwen36-moe --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -np 2 -c 196608 -b 2048 -ub 2048 \
    --mixed-batch-size 1024 --prefill-single-stream --cache-ram 0 --ctx-checkpoints 16 --checkpoint-min-step 2048 -fa on --host 127.0.0.1 --port 8080 > $P/run.id 2>&1 || { S "$(basename $P) start failed"; docker rm -f $CN >/dev/null 2>&1; return 1; }
  timeout 1500 python3 -I - $CN $P $FIX $ntok > $P/client.log 2>&1 <<'PY'
import sys, json, subprocess, time
cn, out, fix, ntok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ids = json.load(open(fix))[:ntok]
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
  [ $prof = 1 ] && python3 -I $N2/28/an28.py $P/trace.csv > $P/an28.txt 2>&1
  S "$(basename $P) $(tail -1 $P/client.log) engaged=$(grep -Ec "$ENG_RE" $P/server.log) $(grep -Eh "$ENG_RE" $P/server.log | tail -1)"
}

# fk1_check_summary <server.log>: FP32-vs-mmq check lines of either kernel (moe-fable-check / moe-fp32-check)
fk1_check_summary() {
  grep -Eh 'moe-(fable|fp32)-check' "$1" | python3 -I -c '
import sys, re
n = bad = 0; mx = 0.0
for l in sys.stdin:
    n += 1
    m = re.search(r"relRMSE(?:\([^)]*\))?[ =:]*([0-9.eE+-]+)", l); mx = max(mx, float(m.group(1))) if m else mx
    f = re.findall(r"(?<!ref_)nonfinite[ =:]*([0-9]+)", l); bad += any(int(v) for v in f) or not f
print(f"check lines {n}, lines with nonfinite>0 {bad}, max relRMSE {mx:.3e}")'
}
