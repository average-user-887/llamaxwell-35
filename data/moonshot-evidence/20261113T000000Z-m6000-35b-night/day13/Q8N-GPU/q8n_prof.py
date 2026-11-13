#!/usr/bin/env python3
"""Q8N nvprof one-decode-step profile (call only from run-q8n-gpu.sh). usage: q8n_prof.py <outdir> <arms e.g. F,Q,K>   env Q8N_GPU
Per arm: server under nvprof (CUDA 12.8 devel image), restore slot 0 (F: D0 f16 file; Q/K: q8 files), decode 12 tokens of the 98000-token prompt, stop.
Analysis: the last 80 ofa_decode_hs_ar* launches = 8 decode steps x 10 layers; reports ofa ms/step and total GPU busy ms in that window."""
import sys, os, json, time, csv, importlib.util
spec = importlib.util.spec_from_file_location('q8n_srv', os.path.dirname(os.path.abspath(__file__)) + '/q8n_srv.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
NV = 'nvidia/cuda:12.8.1-devel-ubuntu24.04'
def launch_prof(arm, name, od):
    args, extra = S.ARM[arm]; env = list(S.ENV) + extra
    sl = (S.QSLOTS + ':/slots') if arm in 'QGK' else (S.D0 + '/slots:/slots:ro')
    cmd = ['docker', 'run', '-d', '--name', name, '-p', '127.0.0.1:%d:8080' % S.PORT, '--device', 'nvidia.com/gpu=' + S.GPU, '-u', '%d:%d' % (os.getuid(), os.getgid()), '-e', 'HOME=/tmp',
           '-v', '<MODELS>:/models:ro', '-v', sl, '-v', S.BIN + ':/candidate:ro', '-v', od + ':/out', '-e', 'LD_LIBRARY_PATH=/candidate']
    for e in env: cmd += ['-e', e]
    cmd += ['--entrypoint', '/usr/local/cuda/bin/nvprof', NV, '--csv', '--print-gpu-trace', '--log-file', '/out/prof.csv', '/candidate/llama-server', '-m', '/models/occamy-1.0-Q4_0-headq40b.gguf',
            '--alias', 'qwen36-moe', '--override-kv', 'tokenizer.ggml.pre=str:qwen35', '-ngl', '999', '-np', '2', '-c', '196608', '-b', '2048', '-ub', '1024', '--cache-ram', '0', '--ctx-checkpoints', '16',
            '--checkpoint-min-step', '2048', '-fa', 'on', '--host', '0.0.0.0', '--port', '8080', '--jinja', '--metrics', '--slot-save-path', '/slots', '-lv', '4'] + args
    r = S.sh(cmd); assert r.returncode == 0, r.stderr
def analyze(path):
    rows = list(csv.reader(open(path, errors='replace'))); h = next(i for i, r in enumerate(rows) if r and r[0] == 'Start'); hd = rows[h]; ui = hd.index('Duration'); ni = hd.index('Name')
    unit = rows[h + 1][ui]; f = {'s': 1e3, 'ms': 1.0, 'us': 1e-3, 'ns': 1e-6}[unit]; data = [r for r in rows[h + 2:] if len(r) > ni]
    ofa = [i for i, r in enumerate(data) if 'ofa_decode_hs_ar' in r[ni]]
    if len(ofa) < 80: return {'error': 'only %d ofa launches' % len(ofa)}
    w = ofa[-80:]; lo = w[0]; ms = sum(float(data[i][ui]) * f for i in w); busy = 0.0
    for r in data[lo:]:
        try: busy += float(r[ui]) * f
        except ValueError: pass
    return {'ofa_launches_total': len(ofa), 'ofa_kernel': data[w[-1]][ni][:90], 'ofa_ms_per_step': ms / 8, 'ofa_ms_per_launch': ms / 80, 'gpu_busy_ms_per_step_window': busy / 8,
            'grid': data[w[-1]][2:5], 'regs': data[w[-1]][8]}
def main():
    out = sys.argv[1]; arms = sys.argv[2].split(','); os.makedirs(out, exist_ok=True); res = {}
    for arm in arms:
        od = os.path.abspath(out + '/' + arm); os.makedirs(od, exist_ok=True); name = 'nq8g-prof-%s-%d' % (arm.lower(), int(time.time()) % 100000)
        if arm in 'QGK' and not all(os.path.exists(S.QSLOTS + '/' + f) for f in S.SQA[arm].values()): res[arm] = {'error': 'no q8 slot files'}; continue
        assert not S.gpu_apps(), 'GPU busy'
        try:
            launch_prof(arm, name, od); t0 = time.time()
            while time.time() - t0 < 900:
                try:
                    if S.call('/health', method='GET', timeout=3)[0] == 200: break
                except Exception: pass
                time.sleep(3)
            else: raise RuntimeError('not ready')
            ids = S.ids_load(); fs = S.SQA[arm] if arm in 'QGK' else S.SF; S.restore(0, fs['s0'])
            o = S.par([(ids[0:98000], 12, 0)]); res[arm] = {'decode': {k: v for k, v in o['s0'].items() if k != 'tokens'}}
        except Exception as e: res[arm] = {'error': repr(e)}
        finally:
            S.sh(['docker', 'stop', '-t', '120', name]); lg = S.sh(['docker', 'logs', name]); open(od + '/server.log', 'w').write((lg.stdout + lg.stderr)[-60000:]); S.sh(['docker', 'rm', '-f', name])
        try: res[arm].update(analyze(od + '/prof.csv'))
        except Exception as e: res[arm]['analyze_error'] = repr(e)
        print(arm, json.dumps(res[arm]), flush=True)
    json.dump(res, open(out + '/prof.json', 'w'), indent=1)
if __name__ == '__main__': main()
