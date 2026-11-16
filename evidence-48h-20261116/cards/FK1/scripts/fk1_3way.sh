#!/usr/bin/env bash
# FK1 head-to-head, one GPU session (NOT run by the coder; the orchestrator queues it). ~75 min GPU hold at ROUNDS=5.
# Arms, all with the prod batch8 server env and the A4 microbench (nvprof, 16384-token single-slot prefill, MoE ms/ubatch):
#   prod  = prod batch8 libs, no flag            (mmq)
#   opus  = A4 libs,  GGML_CUDA_MOE_FP32_PREFILL=1    (note: A4 is built on the bundle9 base, so compare the MoE class, not busy)
#   fable = FK1 libs, GGML_CUDA_MOE_FABLE_PREFILL=1
# (1) correctness per candidate first: test-backend-ops -o MUL_MAT_ID through each kernel (MIN=1), then a 4096-token prefill with
#     the kernel's CHECK=64 (nonfinite and relRMSE vs mmq on the same ops).
# (2) ROUNDS (default 5) interleaved rounds, arm order rotated every round, one server start per row.
# Output: $OUT/3way.md (per-arm median/min/max MoE ms/ubatch, speedup vs prod, engaged lines per row) and 3way_rows.tsv.
# Launch: N3_GPU=... N3_UNLOAD_URL=... nohup bash fk1_3way.sh > launch3.log 2>&1 &
source "$(dirname "$(readlink -f "$0")")/fk1_common.sh"
ROUNDS=${ROUNDS:-5}
OUT=$E/run3-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT/prof; chmod 777 $OUT/prof; echo $OUT > $E/LATEST3
libs_of() { case $1 in prod) echo $LIBS_PROD;; opus) echo $LIBS_A4;; fable) echo $LIBS_FK1;; esac; }
flag_of() { case $1 in prod) echo "";; opus) echo "-e GGML_CUDA_MOE_FP32_PREFILL=1";; fable) echo "-e GGML_CUDA_MOE_FABLE_PREFILL=1";; esac; }
fk1_take_gpu
for f in $LIBS_PROD $LIBS_A4 $LIBS_FK1; do S "lib $(sha256sum $f/libggml-cuda.so | cut -c1-64) $(basename $(dirname $f))/$(basename $f)"; done

fk1_tbo $OUT/tbo-opus  $LIBS_A4  -e GGML_CUDA_MOE_FP32_PREFILL=1  -e GGML_CUDA_MOE_FP32_PREFILL_MIN=1
fk1_tbo $OUT/tbo-fable $LIBS_FK1 -e GGML_CUDA_MOE_FABLE_PREFILL=1 -e GGML_CUDA_MOE_FABLE_PREFILL_MIN=1
fk1_server_run $OUT/check-opus  $LIBS_A4  4096 0 "-e GGML_CUDA_MOE_FP32_PREFILL=1 -e GGML_CUDA_MOE_FP32_CHECK=64"
S "CHECK opus $(fk1_check_summary $OUT/check-opus/server.log)"
fk1_server_run $OUT/check-fable $LIBS_FK1 4096 0 "-e GGML_CUDA_MOE_FABLE_PREFILL=1 -e GGML_CUDA_MOE_FABLE_CHECK=64"
S "CHECK fable $(fk1_check_summary $OUT/check-fable/server.log)"

ORD=(prod opus fable)
for r in $(seq 1 $ROUNDS); do
  for k in 0 1 2; do
    a=${ORD[$(( (k + r - 1) % 3 ))]}
    fk1_server_run $OUT/prof/r$r-$a "$(libs_of $a)" 16384 1 "$(flag_of $a)"
  done
done
python3 -I - $OUT > $OUT/3way.md 2>$OUT/3way.err <<'PY'
import sys, json, glob, re, statistics as st
o = sys.argv[1]; rows = {}; tsv = ['round\tarm\tmoe_ms_ubatch\tbusy_ms_ubatch\tn_ubatch\tengaged_lines']
for d in sorted(glob.glob(o + '/prof/r*-*')):
    rnd, arm = re.match(r'r(\d+)-(\w+)', d.split('/')[-1]).groups()
    try: segs = json.load(open(d + '/trace.csv.segs.json'))
    except Exception as e: tsv.append(f'{rnd}\t{arm}\tVOID\t{e}'); continue
    mid = segs[1:-1] if len(segs) > 3 else segs
    moe = st.median(s['by'].get('moe_mmq', 0) + s['by'].get('moe', 0) for s in mid); busy = st.median(s['busy'] for s in mid)
    eng = len(re.findall(r'moe-(?:fable|fp32): engaged', open(d + '/server.log', errors='replace').read()))
    void = arm != 'prod' and eng == 0
    tsv.append(f'{rnd}\t{arm}\t{moe:.1f}\t{busy:.1f}\t{len(mid)}\t{eng}' + ('\tVOID(not engaged)' if void else ''))
    if not void: rows.setdefault(arm, []).append((moe, busy, eng))
open(o + '/3way_rows.tsv', 'w').write('\n'.join(tsv) + '\n')
p = st.median(m for m, _, _ in rows['prod']) if rows.get('prod') else None
print('| arm | n | MoE ms/ubatch median [min,max] | vs prod | busy ms/ubatch median | engaged lines/row |'); print('|---|---|---|---|---|---|')
for a in ('prod', 'opus', 'fable'):
    v = rows.get(a, [])
    if not v: print(f'| {a} | 0 | VOID | | | |'); continue
    m = [x[0] for x in v]; md = st.median(m)
    print(f"| {a} | {len(v)} | {md:.1f} [{min(m):.1f},{max(m):.1f}] | {(p/md if p else float('nan')):.4f}x | {st.median(x[1] for x in v):.1f} | {min(x[2] for x in v)}..{max(x[2] for x in v)} |")
PY
S "3way: $(grep -E '^\| (prod|opus|fable)' $OUT/3way.md | tr '\n' ' ')"
for a in opus fable; do S "CHECK $a $(fk1_check_summary $OUT/check-$a/server.log)"; done
S done
