#!/usr/bin/env python3
"""Card-2 S2/S4: run the frozen BFCL worker (bfcl_pairs.py, unchanged) through the R3 wrapper run_agentic_r3.py,
loaded from its source with three in-memory patches (shared file is not edited):
  1. --ctx-checkpoints and -ub/-mixed-batch from env QA_CKPT / argv (BFCL recipe otherwise),
  2. wrapper run dir -> /tmp/qa-prefill-card/<sub>/runs (the evaluator output must stay in the prep dir, frozen rule),
  3. stage wall cap for early/preview taken from bfcl_pairs STAGE_SECONDS (wrapper hard-codes 560 s for them).
R3-lossless env from run_bfcl_lossless.sh minus GGML_CUDA_MAXWELL_GRAPHS (system rule); GGML_CUDA_OCCAMY_DEBUG unset.
A thermal watchdog removes only this process's own r3card2-bfcl-* container at >=80 C and marks the run invalid.
usage: s2_bfcl.py STAGE MODE TAG [extra --env K=V ...]"""
import os, subprocess, sys, threading, time, types, argparse, json, re
from pathlib import Path
sys.dont_write_bytecode = True
R3 = Path('~/.gemini/antigravity/brain/5dde1567-112b-432c-a28b-3e2e14940eab/scratch/r3-bench')
WR = R3 / 'card2-agentic/run_agentic_r3.py'
sys.path.insert(0, str(R3)); sys.path.insert(0, str(WR.parent))
import r3_bench as rb  # noqa: E402

STAGE, MODE, TAG = sys.argv[1:4]
EXTRA_ENV = sys.argv[4:]
CKPT = os.environ.get('QA_CKPT', '2')
UB = int(os.environ.get('QA_UB', 1024)); MB = int(os.environ.get('QA_MB', 1024))
SUB = os.environ.get('QA_SUB', 's2')
assert 'GGML_CUDA_MAXWELL_GRAPHS' not in os.environ and 'GGML_CUDA_OCCAMY_DEBUG' not in os.environ

src = WR.read_text()
cap_line = '            max_cap = 36000 if (bench == "bfcl" and stage == "full") else (1500 if (bench == "bfcl" and stage == "smoke") else (50400 if stage == "full" else 560))'
assert src.count(cap_line) == 1
src = src.replace(cap_line, '            max_cap = {"smoke": 1500, "early": 3600, "preview": 9000, "full": 36000}[stage]')
mod = types.ModuleType('run_agentic_r3'); mod.__file__ = str(WR)
exec(compile(src, str(WR), 'exec'), mod.__dict__)
OUTR = Path('/tmp/qa-prefill-card') / SUB / 'runs'; OUTR.mkdir(parents=True, exist_ok=True)
mod.WRAPPER_RUNS_DIR = OUTR

_bsa = rb.build_server_args
def server_args(mode, n_ctx, ub, mb):
    a = _bsa(mode, n_ctx, ub, mb); a[a.index('--ctx-checkpoints') + 1] = CKPT; return a
rb.build_server_args = server_args

ENV = ['LLAMA_MTP_LAST_QUERY=1', 'LLAMA_OCCAMY_COMPACT_DRAFT=1', 'GGML_CUDA_OCCAMY_PERSIST_SCRATCH2=1',
       'GGML_CUDA_OCCAMY_MOE_SH_Q8=1', 'GGML_CUDA_OCCAMY_FA_Q1_R2_FAST=1'] + EXTRA_ENV
args = argparse.Namespace(bench='bfcl', stage=STAGE, mode=MODE, dso_preset='card1', dso_dir=rb.DEFAULT_DSO_DIR, n_ctx=84224,
                          ub=UB, mixed_batch=MB, env=ENV, unset=['GGML_CUDA_OCCAMY_DEBUG', 'GGML_CUDA_MAXWELL_GRAPHS'],
                          pair_env=[], tag=TAG, prewarm=False, cap=int(os.environ.get('QA_CAP', 12000)))

uuid = rb.resolve_gpu_uuid(); me = 'r3card2-bfcl-%s-' % MODE; pid = str(os.getpid())
state = dict(tmax=0, tripped=False); stop = threading.Event()
tlog = OUTR / ('temp-%s-%s-%s.log' % (STAGE, TAG, pid))
PACE_HI = int(os.environ.get('QA_PACE_HI', 77)); PACE_LO = int(os.environ.get('QA_PACE_LO', 72))
PACE_MAX = float(os.environ.get('QA_PACE_MAX', 15)); PACE_GAP = float(os.environ.get('QA_PACE_GAP', 20))
state.update(pauses=0, paused_s=0.0)
def own():
    return [n for n in subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split()
            if n.startswith(me) and n.endswith('-' + pid)]
def gtemp():
    r = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    try: return int(r.stdout.strip().splitlines()[0])
    except Exception: return -1
def wd():
    # pacing (documented): at >=PACE_HI pause own container (docker pause = cgroup freeze) until <=PACE_LO or PACE_MAX s,
    # then >=PACE_GAP s running before the next pause; worker attempt_timeout is 90 s so pauses stay short.
    last_resume = 0.0
    with open(tlog, 'a') as f:
        while not stop.is_set():
            t = gtemp(); state['tmax'] = max(state['tmax'], t); f.write('%d %d\n' % (time.time(), t)); f.flush()
            if t >= 80:
                state['tripped'] = True
                for n in own(): subprocess.run(['docker', 'rm', '-f', n], capture_output=True)
                f.write('TRIP %d C\n' % t); return
            if t >= PACE_HI and time.time() - last_resume >= PACE_GAP:
                names = own()
                if names:
                    t0 = time.time(); subprocess.run(['docker', 'pause'] + names, capture_output=True)
                    while time.time() - t0 < PACE_MAX and not stop.is_set():
                        stop.wait(1.0); t = gtemp(); state['tmax'] = max(state['tmax'], t); f.write('%d %d P\n' % (time.time(), t))
                        if t <= PACE_LO: break
                    subprocess.run(['docker', 'unpause'] + names, capture_output=True); last_resume = time.time()
                    state['pauses'] += 1; state['paused_s'] += last_resume - t0
                    f.write('PAUSE %d %d %.1f\n' % (t0, last_resume, last_resume - t0)); f.flush()
            stop.wait(1.0)
print('admission:', re.sub(r'GPU-[0-9a-fA-F-]+', 'GPU-REDACTED', rb.admit_idle_fleet())[:300], flush=True)
mod.verify_frozen_workers()
th = threading.Thread(target=wd, daemon=True); th.start()
rc = 0
try:
    mod.run_arm(args)
except Exception as ex:
    print('run_arm exception:', str(ex)[:400], flush=True); rc = 1
finally:
    stop.set(); th.join(5)
print('S2DONE stage=%s mode=%s tag=%s ckpt=%s ub=%d tmax=%d tripped=%s pauses=%d paused_s=%.0f rc=%d' % (STAGE, MODE, TAG, CKPT, UB, state['tmax'], state['tripped'], state['pauses'], state['paused_s'], rc), flush=True)
sys.exit(rc or (3 if state['tripped'] else 0))
