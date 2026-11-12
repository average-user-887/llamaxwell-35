import sys, os, json, time, subprocess, urllib.request, pathlib, hashlib, csv, datetime
import numpy as np
W = '<LAB>/codex-investigation-20261112'
FIX = '<LAB>/occamy35-day-20261112/evidence/fixtures/tokens'
sys.dont_write_bytecode = True
sys.path.insert(0, W)
import model_quality_grid as mq   # Codex harness: run_grid / preflight / score (reused unchanged)
IMAGE = 'sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'
LIB_V4 = W + '/prefill/gemm/integration-v4/src/build/bin/libggml-cuda.so'
GPU = os.environ['G1_GPU']
BASE_ENV = 'GGML_CUDA_MOE_DOWNX=1 GGML_CUDA_MOE_DOWNH=2 GGML_CUDA_OCCAMY_GDN_QKREUSE=1 GGML_CUDA_OCCAMY_GDN_PCSHARE=1 GGML_CUDA_MMVQ_SM52=1 GGML_CUDA_RMS_WARP=0 GGML_CUDA_HEAD_ROWPACK_SM52=1 GGML_CUDA_MMQ_COMPACT_SM52=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_NORM_Q8=1 GGML_CUDA_CONCAT_T=1 GGML_CUDA_OCCAMY_GDN_STATE_DIRECT=1 GGML_CUDA_GATE_COPY=1 GGML_CUDA_OCCAMY_GDN=1 GGML_CUDA_MMQ_XMAD_SM52=1 GGML_CUDA_GDN_COLS=2,4 GGML_CUDA_MOE_LOP3=1 GGML_CUDA_MOE_TAILBAL=1 GGML_CUDA_OCCAMY_GDN_PREF=1 GGML_CUDA_OCCAMY_HOSTGAP=1 GGML_CUDA_OCCAMY_SMALLFUSE=2 GGML_CUDA_MMQ_TAIL_SM52=1 GGML_CUDA_OCCAMY_GDN_REPACK=1 GGML_CUDA_OCCAMY_N2SPLIT=1 GGML_CUDA_OCCAMY_N2GDN=1'.split()
FA_ENV = ['GGML_CUDA_OCCAMY_FA=1', 'GGML_CUDA_OCCAMY_FUSED=1', 'GGML_CUDA_OCCAMY_FA_PIPE=3', 'GGML_CUDA_OCCAMY_FA_ADDR=1']

def docs(n):
    fx = json.load(open(W + '/fixture_tokens.json'))
    d0 = json.load(open(FIX + '/d0.json')); d1 = json.load(open(FIX + '/d1.json'))
    u = []
    import glob
    for f in sorted(glob.glob(FIX + '/u[0-9][0-9].json')): u += json.load(open(f))
    L = 65552
    cand = {'d0': d0[:L], 'd1': d1[:L]}
    for k in range(1, 8): cand['fx%d' % k] = fx[k * 98016:k * 98016 + L]
    for k in range(0, 9): cand['u%d' % k] = u[k * L:(k + 1) * L]
    order = ['d0', 'd1', 'fx1', 'u0', 'fx2', 'u1', 'fx3', 'u2', 'fx4', 'u3', 'fx5', 'u4', 'fx6', 'u5', 'fx7', 'u6', 'u7', 'u8']
    out = [(k, cand[k]) for k in order if len(cand[k]) == L]
    return [('codex', fx[:98016])] + out[:n]

class Server:
    def __init__(self, base): self.base = base
    def call(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(body if body is not None else {}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(r, timeout=900) as h: return json.loads(h.read())
    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as h: return h.read()

def main():
    arm, out, minutes, maxdocs, ctx, extra = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6:]
    t_end = time.time() + minutes * 60
    d = out / arm; d.mkdir(parents=True)
    env = list(BASE_ENV) + ([] if arm == 'C' else FA_ENV) + (['GGML_CUDA_FA_MAXWELL_GEMM=1'] if arm in ('B', 'B2') else []) + (['GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2'] if arm == 'Ap' else [])
    name = 'g1-%s-%s' % (arm.lower(), out.name.lower())
    port = '18436'
    args = ['docker', 'create', '--name', name, '--network', 'host', '--restart', 'no', '--device', 'nvidia.com/gpu=' + GPU, '-v', '<MODELS>:/models:ro', '-v', '<LAB>/occamy35-day-20261112/rounds/D0/slots:/slots:ro', '-v', LIB_V4 + ':/app/libggml-cuda.so:ro']
    for e in env: args += ['-e', e]
    flags = ['-m', '/models/occamy-1.0-Q4_0-headq40b.gguf', '--alias', 'g1', '--override-kv', 'tokenizer.ggml.pre=str:qwen35', '-ngl', '999', '-np', '1', '-c', ctx, '-b', '2048', '-ub', '1024', '--cache-ram', '0', '--ctx-checkpoints', '16', '--checkpoint-min-step', '2048', '--host', '127.0.0.1', '--port', port, '--slot-save-path', '/slots', '--jinja', '--metrics', '-lv', '4'] + extra
    if '-fa' not in extra: flags += ['-fa', 'on']
    args += [IMAGE] + flags
    json.dump({'arm': arm, 'env': env, 'flags': flags, 'lib_sha256': hashlib.sha256(open(LIB_V4, 'rb').read()).hexdigest()}, open(d / 'launch.json', 'w'), indent=1)
    subprocess.run(args, check=True, capture_output=True)
    subprocess.run(['docker', 'start', name], check=True, capture_output=True)
    srv = Server('http://127.0.0.1:' + port)
    rows_csv = open(d / 'rows.csv', 'w', newline=''); cw = csv.writer(rows_csv)
    cw.writerow(['doc', 'rep', 'checkpoint', 'row', 'position', 'label', 'label_logprob', 'raw_top1', 'full', 'prompt_n', 'cache_n', 'prompt_s'])
    try:
        t0 = time.time()
        while True:
            try:
                if json.loads(srv.get('/health')).get('status') == 'ok': break
            except Exception: pass
            if time.time() - t0 > 900 or subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], capture_output=True, text=True).stdout.strip() != 'true':
                raise RuntimeError('server did not start')
            time.sleep(3)
        srv.call('/completion', {'prompt': [1, 2, 3, 4, 5, 6, 7, 8], 'n_predict': 8, 'temperature': 0, 'id_slot': 0})
        dl = docs(maxdocs)
        status = open(d / 'progress.log', 'a')
        def log(m): status.write('%s %s\n' % (datetime.datetime.utcnow().strftime('%H:%M:%S'), m)); status.flush()
        pf_tokens = dl[0][1]
        mq.preflight(srv, pf_tokens, lambda n, v: json.dump({k: v[k] for k in v if k not in ('completion_probabilities', 'top_logprobs')} if isinstance(v, dict) else v, open(d / ('pre-' + n), 'w')))
        jobs = ([('codex', 0), ('codexs', 0)] if arm == 'Ap' else [('codex', 0)] if arm in ('A2', 'B2') else [('codex', 0), ('codexs', 0)] + [(k, 0) for k, _ in dl[1:]])
        tok = dict(dl); tok['codexs'] = dl[0][1]
        for k, rep in jobs:
            if rep == 0 and k not in ('codex', 'codexs') and time.time() + 480 > t_end: log('budget stop before ' + k); break
            tokens = tok[k]
            cps = (16384, 32768, 65836) if k == 'codexs' else (16384, 32768, 65536)
            state = {'cur': None}
            def save(n, v):
                if n.startswith('quality-row-'):
                    fv = v['full_vocabulary']
                    if fv:
                        arr = np.full(mq.VOCABULARY, np.nan, dtype=np.float32)
                        for e in v['top_logprobs']: arr[e['id']] = e['logprob']
                        np.save(d / ('%s_r%d_%d.npy' % (k, rep, v['checkpoint'])), arr)
                    cw.writerow([k, rep, v['checkpoint'], v['checkpoint_row_index'], v['position'], v['label'], v['label_logprob'], v['raw_top1'], int(fv), '', '', '']); rows_csv.flush()
                elif n.startswith('prefill-stage'):
                    log('%s r%d stage pos=%d wall=%.1f pps=%.1f' % (k, rep, v['position'], v['wall_s'], v['timings']['prompt_per_second']))
            srv.call('/slots/0?action=erase', {})
            ts = time.time()
            mq.run_grid(srv, tokens[:cps[-1] + 16], save, checkpoints=cps)
            log('done %s rep%d in %.0fs' % (k, rep, time.time() - ts))
    finally:
        rows_csv.close()
        subprocess.run(['docker', 'logs', name], stdout=open(d / 'server.log', 'wb'), stderr=subprocess.STDOUT)
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True)

main()
