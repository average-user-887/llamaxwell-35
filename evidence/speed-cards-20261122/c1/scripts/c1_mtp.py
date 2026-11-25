#!/usr/bin/env python3
"""Speed card 1: two-slot native MTP on/off at 2x64K and 2x96K (N3R). Derived from qa-prefill s1_abba.py.\nOld doc: QA prefill card step 2: two-slot N3R append makespan, arm A (r3-lossless ub1024) vs arm B
(lossless + ub2048 + mixed-batch 1024 + FA_MAXWELL_GEMM_V5=1), order A B B A, one GPU lease.
GPU UUID is resolved by r3_bench and never printed or stored. GGML_CUDA_MAXWELL_GRAPHS and
GGML_CUDA_OCCAMY_DEBUG are removed from both arms. Per-run watchdog kills only this run's container at >=80 C."""
import fcntl, json, os, random, re, subprocess, sys, threading, time
from contextlib import contextmanager
from pathlib import Path
sys.dont_write_bytecode = True
R3 = Path('<SCRATCH>/r3-bench')
sys.path.insert(0, str(R3))
import r3_bench as rb  # noqa: E402

OUT = Path(os.environ.get('QA_OUT', '/tmp/qa-speed-cards/c1'))
B0 = Path('<STORAGE>/m6000-llama-cpp/occamy35-48h-20261115/B0')
BFCL_ENV = R3 / 'card2-agentic/runs/r3-bfcl-full-r3-lossless-native-20261121T013713Z-482738/environment.args'
ORDER = os.environ.get('QA_ORDER', 'ON64,OFF64,OFF64,ON64,ON96,OFF96,OFF96,ON96').split(',')
STOP_AT = float(os.environ.get('QA_STOP_EPOCH', '0'))  # do not start a run after this epoch
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


ARMS = {}
for c in (64, 78, 96):
    ARMS['ON%d' % c] = dict(mode='native', ctx=c * 1024)
    ARMS['OFF%d' % c] = dict(mode='serial', ctx=c * 1024)
CKPT = os.environ.get('QA_CKPT', '8')   # B0: ub1024 + ctx-checkpoints 8, no V5, GRAPHS stripped
UB = 1024


_bsa = rb.build_server_args


def server_args(mode, n_ctx, ub, mb):   # BFCL recipe args, but 2 slots (N3R) and cache-ram 0 as in the BFCL run
    a = _bsa(mode, n_ctx, ub, mb)
    a[a.index('-np') + 1] = '2'
    a[a.index('--cache-ram') + 1] = '0'
    a[a.index('--ctx-checkpoints') + 1] = CKPT
    return a


rb.build_server_args = server_args


def gpu_temp(uuid):
    r = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=temperature.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    try: t, m = r.stdout.strip().splitlines()[0].split(','); gpu_temp.mem = int(m); return int(t)
    except Exception: return -1


PAUSE_HI = int(os.environ.get('QA_PAUSE_HI', 77))   # pacing (all phases, both arms): freeze own container at >= this
PAUSE_LO = int(os.environ.get('QA_PAUSE_LO', 72))   # ... resume at <= this
PAUSE_MAX = float(os.environ.get('QA_PAUSE_MAX', 15))  # ... or after this many seconds


class Watchdog(threading.Thread):
    """80 C kill of own container (run invalid); docker pause/unpause own container for pacing, recorded with timestamps."""
    def __init__(self, uuid, name, log):
        super().__init__(daemon=True); self.uuid, self.name, self.log = uuid, name, log
        self.stop = threading.Event(); self.tmax = 0; self.mmax = 0; self.tripped = False; self.pauses = []
    def pause(self, f, t_hi):
        t0 = time.time(); subprocess.run(['docker', 'pause', self.name], capture_output=True, timeout=20)
        f.write('PAUSE %d C\n' % t_hi); f.flush(); t = t_hi
        while not self.stop.is_set() and time.time() - t0 < PAUSE_MAX:
            self.stop.wait(1.0); t = gpu_temp(self.uuid)
            if 0 <= t <= PAUSE_LO: break
        subprocess.run(['docker', 'unpause', self.name], capture_output=True, timeout=20)
        d = round(time.time() - t0, 1); self.pauses.append((round(t0, 1), d)); f.write('UNPAUSE after %.1fs at %d C\n' % (d, t)); f.flush()
    def run(self):
        with open(self.log, 'a') as f:
            while not self.stop.is_set():
                t = gpu_temp(self.uuid); self.tmax = max(self.tmax, t); m = getattr(gpu_temp, 'mem', 0); self.mmax = max(self.mmax, m)
                f.write('%d %d %d\n' % (int(time.time()), t, m)); f.flush()
                if t >= HARD_C:
                    self.tripped = True
                    subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True, timeout=20)
                    f.write('TRIP %d C, removed own container\n' % t); return
                if t >= PAUSE_HI: self.pause(f, t)
                self.stop.wait(2.0)


def cool(uuid, log):
    t0 = time.time()
    while True:
        t = gpu_temp(uuid)
        if 0 <= t <= COOL_C or time.time() - t0 > 1800: break
        time.sleep(15)
    log('cooldown %.0fs -> %d C' % (time.time() - t0, t))


@contextmanager
def lease_blocking(log):
    # per-run blocking flock on the shared bench lock (other workers queue on the same lock)
    fd = os.open(str(rb.LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    t0 = time.time(); fcntl.flock(fd, fcntl.LOCK_EX)
    log('lease acquired after %.0fs wait' % (time.time() - t0))
    try: yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd); log('lease released')


def main():
    OUT.mkdir(parents=True, exist_ok=True); steps = OUT / 'steps.log'
    def log(s):
        line = '%s %s' % (time.strftime('%H:%M:%S'), re.sub(r'GPU-[0-9a-fA-F-]+', 'GPU-REDACTED', s)); print(line, flush=True)
        with open(steps, 'a') as f: f.write(line + '\n')
    dso = rb.resolve_dso_map('card1', rb.DEFAULT_DSO_DIR)
    uuid = rb.resolve_gpu_uuid()
    results = []
    log('order=%s' % ','.join(ORDER))
    for i, arm in enumerate(ORDER, 1):
        if STOP_AT and time.time() > STOP_AT: log('stop time reached, not starting run %d %s' % (i, arm)); break
        with lease_blocking(log):
            if STOP_AT and time.time() > STOP_AT: log('stop time reached after lease wait, not starting run %d %s' % (i, arm)); break
            try: log('fleet admission: %s' % rb.admit_idle_fleet()[:300].replace('\n', ' '))
            except Exception as ex: log('fleet admission FAILED: %s' % ex); raise
            cool(uuid, log)
            cfg = ARMS[arm]; env = base_env(); CTX = cfg['ctx']
            od = OUT / ('%d_%s' % (i, arm)); od.mkdir(parents=True, exist_ok=True)
            sess = rb.ServerSession(dso, od, cfg['mode'], 2 * CTX, UB, UB, env, health_timeout=300)
            sess.name = 'r3bench-c1-%s-%d-%d' % (arm.lower(), int(time.time()), random.randint(1000, 9999))
            wd = None; rc = None; t_run = time.time()
            try:
                with sess:
                    wd = Watchdog(uuid, sess.name, od / 'temp.log'); wd.start()
                    e = dict(os.environ, N3R_LOCK_HELD='1', N3R_OUT=str(od / 'n3r')); e.pop('N3_GPU', None)
                    r = subprocess.run(['bash', str(B0 / 'run_n3r.sh'), 'http://127.0.0.1:%d' % sess.port, 'qa-%s' % arm,
                                        '--ctx', str(CTX), '--temp', '0', '--n-predict', '160'], env=e, capture_output=True, text=True, timeout=7200)
                    rc = r.returncode; (od / 'n3r.out').write_text(r.stdout[-20000:] + r.stderr[-5000:])
            except Exception as ex:
                log('run %d %s exception: %s' % (i, arm, str(ex)[:300]))
                for fn in ('server-crash.log', 'server-timeout.log', 'server.log'):
                    fp = od / fn
                    if fp.exists():
                        tail = fp.read_text(errors='replace').strip().splitlines()[-12:]
                        log('%s tail:\n    ' % fn + '\n    '.join(re.sub(r'GPU-[0-9a-fA-F-]+', 'GPU-REDACTED', l)[:220] for l in tail))
                        break
                else:
                    r2 = subprocess.run(['docker', 'logs', '--tail', '12', sess.name], capture_output=True, text=True, timeout=15)
                    log('docker logs tail (%s):\n    %s' % (sess.name, '\n    '.join((r2.stdout + r2.stderr).splitlines()[-12:])[:2500]))
            finally:
                if wd: wd.stop.set(); wd.join(10)
            rj = od / 'n3r' / 'run.json'; rec = dict(run=i, arm=arm, rc=rc, wall_s=round(time.time() - t_run, 1),
                                                     tmax=wd.tmax if wd else None, pauses=wd.pauses if wd else None, vram_peak_mib=wd.mmax if wd else None, ctx=CTX, ckpt=CKPT, invalid=bool(wd and (wd.tripped or wd.tmax >= HARD_C)))
            if rj.exists():
                R = json.loads(rj.read_text())
                rec.update(makespan_s=R.get('makespan_s'), ok=R.get('ok'), n_void=R.get('n_void'), run_sha=R.get('run_sha'),
                           setup_s=[round(s['s'], 1) for s in R.get('setup', [])],
                           turns=[dict(a=t.get('agent'), k=t.get('turn'), prompt_n=t.get('prompt_n'), cache_n=t.get('cache_n'),
                                       prompt_ms=t.get('prompt_ms'), predicted_ms=t.get('predicted_ms'), predicted_n=t.get('predicted_n'), timings=t.get('timings'), sha=t.get('token_sha')) for t in R.get('turns_rec', [])])
            slog = od / 'server.log'
            if slog.exists():
                txt = slog.read_text(errors='replace')
                rec['counters'] = [l.strip()[:300] for l in txt.splitlines() if 'occamy_fa_gemm' in l][-4:]
                rec['cuda_errors'] = len(rb.CUDA_ERROR_RE.findall(txt))
                rec['bufs'] = [l.strip()[-80:] for l in txt.splitlines() if 'buffer size' in l and 'CUDA0' in l]
                rec['draft_lines'] = [l.strip()[-160:] for l in txt.splitlines() if 'draft acceptance' in l or 'n_draft' in l][-4:]
                rec['oom'] = 'out of memory' in txt
            results.append(rec)
            log('run %d %s pauses=%s pause_s=%s vram_peak=%s rc=%s makespan=%s setup=%s ok=%s void=%s tmax=%s invalid=%s sha=%s' % (
                i, arm, len(rec.get('pauses') or []), round(sum(p[1] for p in rec.get('pauses') or []), 1), rec.get('vram_peak_mib'), rc, rec.get('makespan_s'), rec.get('setup_s'), rec.get('ok'), rec.get('n_void'), rec.get('tmax'), rec['invalid'], (rec.get('run_sha') or '')[:12]))
            (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    log('done; analyse with c1_gate.py')


if __name__ == '__main__':
    main()
