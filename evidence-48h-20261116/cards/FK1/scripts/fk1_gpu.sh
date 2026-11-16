#!/usr/bin/env bash
# FK1 MoE Fable prefill GPU checks (NOT run by the coder; the orchestrator queues it). ~35 min GPU hold, +~35 min if the gate passes.
# (1) test-backend-ops -o MUL_MAT_ID, FK1 libs, GGML_CUDA_MOE_FABLE_PREFILL=1 with _MIN=1 (every eligible case through the kernel);
#     'moe-fable: engaged' lines must be >0 and FAIL lines 0.
# (2) Same microbench as A4: nvprof in-model profile, 16384-token single-slot prefill (8 ubatches of 2048), prod batch8 env.
#     Arms: prod libs (batch8) = off, FK1 libs flag off = off-fk1 (control: must match off), FK1 libs flag on = on.
#     Metric: MoE ms per ubatch (an28 classes moe_mmq+moe, median of ubatches 1..6). Gate: off/on >= 1.07x.
# (3) FP32-vs-mmq check: FK1 libs, flag on, GGML_CUDA_MOE_FABLE_CHECK=64, 4096-token prefill (relRMSE expected ~5e-3..2e-2 = mmq q8_1
#     error, nonfinite 0).
# (4) Only if (2) passes: N3 short replay ABBA x3 (n=6 per arm). A = prod libs, B = FK1 libs + flag. A B arm without an engaged line is VOID.
# Launch: N3_GPU=... N3_UNLOAD_URL=... nohup bash fk1_gpu.sh > launch.log 2>&1 &   (FK1_FORCE_N3=1 runs (4) regardless of the gate)
source "$(dirname "$(readlink -f "$0")")/fk1_common.sh"
OUT=$E/run-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/prof; chmod 777 $OUT/prof; echo $OUT > $E/LATEST
FL="-e GGML_CUDA_MOE_FABLE_PREFILL=1"
fk1_take_gpu

fk1_tbo $OUT/tbo $LIBS_FK1 $FL -e GGML_CUDA_MOE_FABLE_PREFILL_MIN=1

fk1_server_run $OUT/prof/off     $LIBS_PROD 16384 1 ""
fk1_server_run $OUT/prof/on      $LIBS_FK1  16384 1 "$FL"
fk1_server_run $OUT/prof/off-fk1 $LIBS_FK1  16384 1 ""
python3 -I - $OUT/prof > $OUT/prof/moe_gate.txt 2>&1 <<'PY'
import sys, json, statistics as st
o = sys.argv[1]; r = {}
for t in ('off', 'off-fk1', 'on'):
    segs = json.load(open(f'{o}/{t}/trace.csv.segs.json'))
    mid = segs[1:-1] if len(segs) > 3 else segs
    r[t] = (st.median(s['by'].get('moe_mmq', 0) + s['by'].get('moe', 0) for s in mid), st.median(s['busy'] for s in mid), len(mid))
    print(f"{t}: MoE {r[t][0]:.1f} ms/ubatch, busy {r[t][1]:.1f} ms/ubatch, n={r[t][2]}")
print(f"control off/off-fk1 {r['off'][0]/r['off-fk1'][0]:.4f}x")
x = r['off'][0]/r['on'][0]
print(f"MoE speedup off/on {x:.4f}x (gate >=1.07) {'PASS' if x >= 1.07 else 'FAIL'}, busy speedup {r['off'][1]/r['on'][1]:.4f}x")
PY
S "moe gate: $(tail -1 $OUT/prof/moe_gate.txt)"

fk1_server_run $OUT/check $LIBS_FK1 4096 0 "$FL -e GGML_CUDA_MOE_FABLE_CHECK=64"
S "CHECK $(fk1_check_summary $OUT/check/server.log)"

if ! grep -q 'gate >=1.07) PASS' $OUT/prof/moe_gate.txt && [ "${FK1_FORCE_N3:-0}" != 1 ]; then S "gate not passed; N3 skipped"; S done; exit 0; fi
BASE="LLAMA_PREFILL_SINGLE_STREAM=1 GGML_CUDA_OCCAMY_DEBUG=1"
i=0
for a in A B B A A B B A A B B A; do
  i=$((i+1)); n=${a}_$i; mkdir -p $OUT/$n
  if [ $a = A ]; then LIBS=$LIBS_PROD; X="$BASE"; else LIBS=$LIBS_FK1; X="$BASE GGML_CUDA_MOE_FABLE_PREFILL=1"; fi
  export N3_EXTRA_ENV="$X" N3_CAND_DIR=$LIBS N3B_GEMM_LIB=$LIBS/libggml-cuda.so N3_BATCH=2048 N3_CKPT=16 N3_CKPT_STEP=2048
  FULL_UB=2048 N3_CAP=1024 N3_BLOCK_ORDERS="prod" python3 -I $D/n3_replay.py $OUT/$n --no-sweep --no-ctxfull > $OUT/$n/run.log 2>&1; S "$n rc=$? libs=$(basename $LIBS)"
  python3 -I $D/n3_analyze.py $OUT/$n > $OUT/$n/analyze.log 2>&1
  L=$(ls $OUT/$n/runs/*/server.log 2>/dev/null)
  ENG=$(grep -Eh "$ENG_RE" $L /dev/null | wc -l)
  S "$n engaged lines=$ENG $(grep -h 'makespan total s' $OUT/$n/analyze.log | head -1)"
  [ $a = B ] && [ "$ENG" = 0 ] && S "$n VOID (flag not engaged)"
done
python3 -I - $OUT > $OUT/n3_abba.txt 2>&1 <<'PY'
import sys, re, glob, statistics as st
o = sys.argv[1]; r = {'A': [], 'B': []}
for f in sorted(glob.glob(o + '/[AB]_*/analyze.log')):
    m = re.search(r'makespan total s \| ([0-9.]+)', open(f).read())
    if m: r[f.split('/')[-2][0]].append(float(m.group(1)))
for k in r: print(k, 'n', len(r[k]), 'mean', round(st.mean(r[k]), 2) if r[k] else None, r[k])
if r['A'] and r['B']: print(f"N3 makespan B vs A {100*(st.mean(r['B'])/st.mean(r['A']) - 1):+.2f}% (negative = faster)")
PY
S "n3: $(tail -1 $OUT/n3_abba.txt)"
S done
