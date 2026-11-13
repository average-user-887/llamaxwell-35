#!/usr/bin/env python3
"""Q8N exactness arm runner (call only from run-q8n-gates.sh, which holds the GPU lock). G1c method, Q8N lib.
usage: q8n_rows_harness.py ARM OUT DOC_LIST      env Q8N_GPU (required, never stored); optional Q8N_BUCKETS, Q8N_NR, Q8N_PER, Q8N_CUT_EPOCH
Arms (same lib = Q8N build, same schedule, same model; only env / KV types differ):
  G   q8_0/q8_0, occamy FA env, KVQ8 unset          -> stock generic q8 FA (reference)
  N   q8_0/q8_0, + GGML_CUDA_OCCAMY_KVQ8=1           -> native occamy q8 read (candidate)
  Gp  G + GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2        -> noise floor of the reference (other split-K reduction order)
  F   f16/f16, occamy FA (prod path)                 -> q8-vs-F16 cache context (informational)
  X   q8_0/q8_0, OCCAMY_FA=2 + KVQ8=1                -> generic result used, native run on the SAME cache, per-call diff logged
  Nk  q8_0/f16 + KVQ8=1 ; Nv f16/q8_0 + KVQ8=1      -> mixed instances <1,0>/<0,1> (optional; compared to F)
Doc k = fixture tokens [k*98368,(k+1)*98368). Per doc: prefill 0..B, NR rows at B..B+NR-1, continue, for B in buckets (shallow + deep)."""
import sys, os, json, time, subprocess, hashlib, pathlib
sys.dont_write_bytecode = True
M = '<LAB>'; W = M + '/codex-investigation-20261112'
Q = M + '/occamy35-night-20261113/Q8N'; GD = Q + '/gates'; BIN = Q + '/build/bin'
IMG = 'sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'   # prod / G1b image (runs the gcc-14 build)
MODEL = '/models/occamy-1.0-Q4_0-headq40b.gguf'
DOCLEN = 98368; BUCKETS = tuple(int(x) for x in os.environ.get('Q8N_BUCKETS', '4096,32768,98304').split(',')); NR = int(os.environ.get('Q8N_NR', 64))
BASE_ENV = [l.split(' ', 1)[1] for l in subprocess.run(['grep', '-o', '-e', '-e GGML_[A-Z0-9_]*=[^ ]*', M + '/run-35b-bundle3-20261112.sh'], capture_output=True, text=True).stdout.split('\n') if l]
assert BASE_ENV and not any('MAXWELL_GRAPHS' in e or 'MAXWELL_GEMM' in e for e in BASE_ENV), 'bad prod env'
assert 'GGML_CUDA_OCCAMY_FA_PIPE=3' in BASE_ENV and 'GGML_CUDA_OCCAMY_FA_ADDR=1' in BASE_ENV and 'GGML_CUDA_OCCAMY_FA=1' in BASE_ENV
assert 'GGML_CUDA_MAXWELL_GRAPHS' not in os.environ and 'GGML_CUDA_FA_MAXWELL_GEMM' not in os.environ, 'forbidden flag in caller env'
ARMS = {'G': ('q8_0', 'q8_0', []), 'N': ('q8_0', 'q8_0', ['GGML_CUDA_OCCAMY_KVQ8=1']), 'Gp': ('q8_0', 'q8_0', ['GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2']),
        'F': ('f16', 'f16', []), 'X': ('q8_0', 'q8_0', ['GGML_CUDA_OCCAMY_KVQ8=1', 'GGML_CUDA_OCCAMY_FA=2']),
        'Gk': ('q8_0', 'f16', []), 'Nk': ('q8_0', 'f16', ['GGML_CUDA_OCCAMY_KVQ8=1']), 'Nv': ('f16', 'q8_0', ['GGML_CUDA_OCCAMY_KVQ8=1'])}
def arm_env(arm):
    kt, vt, extra = ARMS[arm]; env = list(BASE_ENV)
    for x in extra:
        k = x.split('=')[0]; env = [e for e in env if not e.startswith(k + '=')] + [x]
    return kt, vt, env
def events(docs, buckets):
    ev = []
    for k in docs:
        ev += ['T/t/doc%d.i32' % k, 'K']; pos = 0
        for B in buckets:
            ev += ['Odoc%d_b%d' % (k, B), 'P%d:%d' % (pos, B), 'R%d' % NR]; pos = B + NR
    return ev
def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
def run_chunk(arm, out, docs, ctx, tag):
    kt, vt, env = arm_env(arm); od = pathlib.Path(out) / arm; od.mkdir(parents=True, exist_ok=True)
    name = 'nq8g-rows-%s-%s' % (arm.lower(), tag)
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '-u', '%d:%d' % (os.getuid(), os.getgid()), '--device', 'nvidia.com/gpu=' + os.environ['Q8N_GPU'],
           '-v', '<MODELS>:/models:ro', '-v', str(od / 'tok') + ':/t:ro', '-v', str(od) + ':/o', '-v', BIN + ':/app:ro',
           '-v', GD + '/q8n_rows:/app2/q8n_rows:ro', '-e', 'LD_LIBRARY_PATH=/app']
    for e in env: cmd += ['-e', e]
    cmd += ['-w', '/app', '--entrypoint', '/app2/q8n_rows', IMG, MODEL, '/o', str(ctx), '1', kt, vt, '1024'] + events(docs, BUCKETS)
    lib = BIN + '/libggml-cuda.so'
    (od / ('launch_%s.json' % tag)).write_text(json.dumps({'arm': arm, 'k': kt, 'v': vt, 'env': env, 'ctx': ctx, 'docs': docs, 'buckets': BUCKETS, 'nr': NR,
        'lib_sha256': hashlib.sha256(open(lib, 'rb').read()).hexdigest(), 'image': IMG}, indent=1))
    t0 = time.time(); r = sh(cmd); log = r.stdout + r.stderr; (od / ('log_%s.txt' % tag)).write_text(log)
    want = 'kvq8 on' if 'GGML_CUDA_OCCAMY_KVQ8=1' in env else None
    if want and want not in log: print('arm %s: native path NOT active (no "%s" in log)' % (arm, want), flush=True); return 99, time.time() - t0
    if not want and 'kvq8 on' in log: print('arm %s: kvq8 unexpectedly on' % arm, flush=True); return 98, time.time() - t0
    return r.returncode, time.time() - t0
def main():
    arm, out, docs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
    ctx = 98560 if max(BUCKETS) > 66000 else 34000; cut = float(os.environ.get('Q8N_CUT_EPOCH', 0))
    od = pathlib.Path(out) / arm; (od / 'tok').mkdir(parents=True, exist_ok=True)
    import numpy as np
    fx = json.load(open(W + '/fixture_tokens.json'))
    for k in docs: np.asarray(fx[k * DOCLEN:(k + 1) * DOCLEN], dtype=np.int32).tofile(str(od / 'tok' / ('doc%d.i32' % k)))
    per = int(os.environ.get('Q8N_PER', 2)); chunks = [docs[i:i + per] for i in range(0, len(docs), per)]; rcs = 0
    for i, ch in enumerate(chunks):
        if cut and time.time() > cut: print('cut reached; skipping docs', ch, flush=True); break
        rc, dt = run_chunk(arm, out, ch, ctx, 'c%d' % i); print('arm %s docs %s rc=%d %.0fs' % (arm, ch, rc, dt), flush=True)
        if rc != 0: rcs = rc; print('FAILED chunk', ch, flush=True); sh(['docker', 'rm', '-f', 'nq8g-rows-%s-c%d' % (arm.lower(), i)])
    sys.exit(1 if rcs else 0)
if __name__ == '__main__': main()
