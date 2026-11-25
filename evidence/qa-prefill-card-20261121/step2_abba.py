#!/usr/bin/env python3
"""QA prefill card step 2: two-slot N3R append makespan, arm A (r3-lossless ub1024) vs arm B
(lossless + ub2048 + mixed-batch 1024 + FA_MAXWELL_GEMM_V5=1), order A B B A, one GPU lease.
GPU UUID is resolved by r3_bench and never printed or stored. GGML_CUDA_MAXWELL_GRAPHS and
GGML_CUDA_OCCAMY_DEBUG are removed from both arms. Per-run watchdog kills only this run's container at >=80 C."""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode = True
R3 = Path('<SCRATCH>/r3-bench')
sys.path.insert(0, str(R3))
import r3_bench as rb  # noqa: E402

OUT = Path(os.environ.get('QA_OUT', '/tmp/qa-prefill-card/step2'))
B0 = Path('<STORAGE>/m6000-llama-cpp/occamy35-48h-20261115/B0')
BFCL_ENV = R3 / 'card2-agentic/runs/r3-bfcl-full-r3-lossless-native-20261121T013713Z-482738/environment.args'
ORDER = os.environ.get('QA_ORDER', 'A,B,B,A').split(',')
COOL_C = int(os.environ.get('QA_COOL_C', 55))     # pacing: start a run only at or below this
HARD_C = 80                                       # owner hard limit
assert 'GGML_CUDA_MAXWELL_GRAPHS' not in os.environ and 'GGML_CUDA_OCCAMY_DEBUG' not in os.environ


def base_env():
    pairs = []
    for l in BFCL_ENV.read_text().splitlines():
        if '=' not in l: continue
        k, v = l.split('=', 1)
        if k in ('GGML_CUDA_MAXWELL_GRAPHS', 'GGML_CUDA_OCCAMY_DEBUG'): continue
        pairs.append((k, v))
    return pairs


ARMS = {'A': dict(ub=1024, mb=1024, extra=[]),
        'B': dict(ub=2048, mb=1024, extra=[('GGML_CUDA_FA_MAXWELL_GEMM_V5', '1')])}


_bsa = rb.build_server_args


def server_args(mode, n_ctx, ub, mb):   # BFCL recipe args, but 2 slots (N3R) and cache-ram 0 as in the BFCL run
    a = _bsa(mode, n_ctx, ub, mb)
    a[a.index('-np') + 1] = '2'
    a[a.index('--cache-ram') + 1] = '0'
    return a


rb.build_server_args = server_args


def gpu_temp(uuid):
    r = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    try: return int(r.stdout.strip().splitlines()[0])
    except Exception: return -1


class Watchdog(threading.Thread):
    def __init__(self, uuid, name, log):
        super().__init__(daemon=True); self.uuid, self.name, self.log = uuid, name, log
        self.stop = threading.Event(); self.tmax = 0; self.tripped = False
    def run(self):
        with open(self.log, 'a') as f:
            while not self.stop.is_set():
                t = gpu_temp(self.uuid); self.tmax = max(self.tmax, t)
                f.write('%d %d\n' % (int(time.time()), t)); f.flush()
                if t >= HARD_C:
                    self.tripped = True
                    subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True, timeout=20)
                    f.write('TRIP %d C, removed own container\n' % t); return
                self.stop.wait(2.0)


def cool(uuid, log):
    t0 = time.time()
    while True:
        t = gpu_temp(uuid)
        if 0 <= t <= COOL_C or time.time() - t0 > 1800: break
        time.sleep(15)
    log('cooldown %.0fs -> %d C' % (time.time() - t0, t))


def main():
    OUT.mkdir(parents=True, exist_ok=True); steps = OUT / 'steps.log'
    def log(s):
        line = '%s %s' % (time.strftime('%H:%M:%S'), re.sub(r'GPU-[0-9a-fA-F-]+', 'GPU-REDACTED', s)); print(line, flush=True)
        with open(steps, 'a') as f: f.write(line + '\n')
    dso = rb.resolve_dso_map('card1', rb.DEFAULT_DSO_DIR)
    uuid = rb.resolve_gpu_uuid()
    results = []
    with rb.gpu_lease(int(time.time()) + 5 * 3600):
        log('lease held; order=%s' % ','.join(ORDER))
        for i, arm in enumerate(ORDER, 1):
            cool(uuid, log)
            cfg = ARMS[arm]; env = base_env() + cfg['extra']
            od = OUT / ('%d_%s' % (i, arm)); od.mkdir(parents=True, exist_ok=True)
            sess = rb.ServerSession(dso, od, 'native', 168448, cfg['ub'], cfg['mb'], env, health_timeout=300)
            wd = None; rc = None; t_run = time.time()
            try:
                with sess:
                    wd = Watchdog(uuid, sess.name, od / 'temp.log'); wd.start()
                    e = dict(os.environ, N3R_LOCK_HELD='1', N3R_OUT=str(od / 'n3r')); e.pop('N3_GPU', None)
                    r = subprocess.run(['bash', str(B0 / 'run_n3r.sh'), 'http://127.0.0.1:%d' % sess.port, 'qa-%s' % arm,
                                        '--ctx', '84224', '--temp', '0', '--n-predict', '160'], env=e, capture_output=True, text=True, timeout=7200)
                    rc = r.returncode; (od / 'n3r.out').write_text(r.stdout[-20000:] + r.stderr[-5000:])
            except Exception as ex:
                log('run %d %s exception: %s' % (i, arm, str(ex)[:300]))
            finally:
                if wd: wd.stop.set(); wd.join(10)
            rj = od / 'n3r' / 'run.json'; rec = dict(run=i, arm=arm, rc=rc, wall_s=round(time.time() - t_run, 1),
                                                     tmax=wd.tmax if wd else None, invalid=bool(wd and (wd.tripped or wd.tmax >= HARD_C)))
            if rj.exists():
                R = json.loads(rj.read_text())
                rec.update(makespan_s=R.get('makespan_s'), ok=R.get('ok'), n_void=R.get('n_void'), run_sha=R.get('run_sha'),
                           setup_s=[round(s['s'], 1) for s in R.get('setup', [])],
                           turns=[dict(a=t.get('agent'), k=t.get('turn'), prompt_n=t.get('prompt_n'), cache_n=t.get('cache_n'),
                                       prompt_ms=t.get('prompt_ms'), predicted_ms=t.get('predicted_ms'), sha=t.get('token_sha')) for t in R.get('turns_rec', [])])
            slog = od / 'server.log'
            if slog.exists():
                txt = slog.read_text(errors='replace')
                rec['counters'] = [l.strip()[:300] for l in txt.splitlines() if 'occamy_fa_gemm' in l][-4:]
                rec['cuda_errors'] = len(rb.CUDA_ERROR_RE.findall(txt))
            results.append(rec)
            log('run %d %s rc=%s makespan=%s setup=%s ok=%s void=%s tmax=%s invalid=%s sha=%s' % (
                i, arm, rc, rec.get('makespan_s'), rec.get('setup_s'), rec.get('ok'), rec.get('n_void'), rec.get('tmax'), rec['invalid'], (rec.get('run_sha') or '')[:12]))
            (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    log('lease released')
    def mean(a, k): v = [r[k] for r in results if r['arm'] == a and r.get(k) is not None and not r['invalid'] and r.get('ok')]; return (sum(v) / len(v), len(v)) if v else (None, 0)
    (ma, na), (mb, nb) = mean('A', 'makespan_s'), mean('B', 'makespan_s')
    if ma and mb:
        g = (ma - mb) / ma * 100
        log('GATE makespan A=%.2f (n=%d) B=%.2f (n=%d) gain=%.2f%% -> %s' % (ma, na, mb, nb, g, 'PASS' if g >= 1.0 else 'FAIL'))
    else:
        log('GATE makespan: insufficient valid runs')


if __name__ == '__main__':
    main()
