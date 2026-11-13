#!/usr/bin/env python3
"""Q8N server gates (call only from run-q8n-gates.sh, which holds the GPU lock). Q8N lib in every arm; prod BUNDLE3 env + launch shape.
usage: q8n_srv.py <outdir> <order e.g. FQQF,QFFQ,FQQF> [--smoke]        env Q8N_GPU (never stored)
Arms: F = f16 KV (flag-off Q8N lib == BUNDLE3 SASS), Q = -ctk q8_0 -ctv q8_0 + GGML_CUDA_OCCAMY_KVQ8=1 (native), G = q8_0 KV, KVQ8 unset (generic, optional).
Slot state = D0 shape on D0 tokens (deep_text tokenized by the server): s0 = ids[0:98000] (slot 0), s1 = ids[200000:290000] (slot 1).
  F restores the D0 F16 files read-only. F16 files cannot be restored into a q8 cache, so the first q8 session builds q8 files by prefill
  exactly as D0 did (s0: n=24 then save; s1: n=1 then save) into gates/slots and later q8 sessions restore them. Build timings are logged.
Units per session (consecutive same-arm positions share one server): both_deep (slot0 98000 + slot1 90000 cached, n=48 concurrently),
  single_98k (slot0 98000 cached, slot1 erased, n=48), append_4k (slot1 90000 restored + 4096 new tokens, n=1, prefill A/B). Peak VRAM per unit and session."""
import sys, os, json, time, subprocess, threading, http.client, datetime
M = '<LAB>'; Q = M + '/occamy35-night-20261113/Q8N'; GD = Q + '/gates'; BIN = Q + '/build/bin'
D0 = M + '/occamy35-day-20261112/rounds/D0'; IMG = 'sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'
PORT = 18497; GPU = os.environ['Q8N_GPU']; QSLOTS = GD + '/slots'; IDS = GD + '/cache/d0_ids.json'
SF = {'s0': 's0_98k_off0.bin', 's1': 's1_90k_off200000.bin'}; SQ = {'s0': 's0_98k_off0_q8.bin', 's1': 's1_90k_off200000_q8.bin'}
ENV = [l.split(' ', 1)[1] for l in subprocess.run(['grep', '-o', '-e', '-e GGML_[A-Z0-9_]*=[^ ]*', M + '/run-35b-bundle3-20261112.sh'], capture_output=True, text=True).stdout.split('\n') if l]
assert ENV and not any('MAXWELL_GRAPHS' in e or 'MAXWELL_GEMM' in e for e in ENV) and 'GGML_CUDA_MAXWELL_GRAPHS' not in os.environ and 'GGML_CUDA_FA_MAXWELL_GEMM' not in os.environ
ARM = {'F': ([], []), 'Q': (['-ctk', 'q8_0', '-ctv', 'q8_0'], ['GGML_CUDA_OCCAMY_KVQ8=1']), 'G': (['-ctk', 'q8_0', '-ctv', 'q8_0'], [])}
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
def gpu_apps(): return sh(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip()
def call(path, body=None, method='POST', timeout=3600):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'}); r = c.getresponse(); d = r.read(); c.close()
    try: return r.status, json.loads(d)
    except Exception: return r.status, {'raw': d[:300].decode(errors='replace')}
class VRAM(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.peak = 0; s.stop = False
    def run(s):
        while not s.stop:
            o = sh(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used', '--format=csv,noheader,nounits']).stdout.strip()
            if o.isdigit(): s.peak = max(s.peak, int(o))
            time.sleep(0.5)
def launch(arm, name):
    args, extra = ARM[arm]; env = list(ENV) + extra
    sl = (QSLOTS + ':/slots') if arm in 'QG' else (D0 + '/slots:/slots:ro')
    cmd = ['docker', 'run', '-d', '--name', name, '-p', '127.0.0.1:%d:8080' % PORT, '--device', 'nvidia.com/gpu=' + GPU, '-u', '%d:%d' % (os.getuid(), os.getgid()),
           '-v', '<MODELS>:/models:ro', '-v', sl, '-v', BIN + ':/candidate:ro', '-e', 'LD_LIBRARY_PATH=/candidate']
    for e in env: cmd += ['-e', e]
    cmd += ['--entrypoint', '/candidate/llama-server', IMG, '-m', '/models/occamy-1.0-Q4_0-headq40b.gguf', '--alias', 'qwen36-moe', '--override-kv', 'tokenizer.ggml.pre=str:qwen35',
            '-ngl', '999', '-np', '2', '-c', '196608', '-b', '2048', '-ub', '1024', '--cache-ram', '0', '--ctx-checkpoints', '16', '--checkpoint-min-step', '2048', '-fa', 'on',
            '--host', '0.0.0.0', '--port', '8080', '--jinja', '--metrics', '--slot-save-path', '/slots', '-lv', '4'] + args
    r = sh(cmd); assert r.returncode == 0, r.stderr
    return {'arm': arm, 'env_extra': extra, 'args': args}
def comp(ids, n, slot, o, key):
    st, j = call('/completion', {'prompt': ids, 'n_predict': n, 'id_slot': slot, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True})
    t = j.get('timings', {}); o[key] = {k: t.get(k) for k in ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_per_second')}; o[key]['http'] = st
def par(jobs):
    o = {}; th = [threading.Thread(target=comp, args=(*j, o, j[2] and 's1' or 's0')) for j in jobs]; t0 = time.time(); [x.start() for x in th]; [x.join() for x in th]
    o['wall_s'] = time.time() - t0; return o
def restore(slot, fn):
    st, j = call('/slots/%d?action=restore' % slot, {'filename': fn}); assert st == 200, (slot, fn, st, j); return j.get('n_restored')
def ids_load():
    if os.path.exists(IDS): return json.load(open(IDS))
    t = open(D0 + '/deep_text.txt').read(); ids = []
    for i in range(0, len(t), 400000): ids += call('/tokenize', {'content': t[i:i + 400000]})[1]['tokens']   # same chunking as D0 deep.py
    os.makedirs(os.path.dirname(IDS), exist_ok=True); json.dump(ids, open(IDS, 'w')); return ids
def build_q8_slots(ids, log):
    v = VRAM(); v.start(); r = {'unit': 'build_q8_slots', 'utc': now()}
    r['s0_prefill'] = par([(ids[0:98000], 24, 0)]); r['s0_save'] = call('/slots/0?action=save', {'filename': SQ['s0']})[1]
    r['s1_prefill'] = par([(ids[200000:290000], 1, 1)]); r['s1_save'] = call('/slots/1?action=save', {'filename': SQ['s1']})[1]
    v.stop = True; r['vram_peak_mib'] = v.peak; log.write(json.dumps(r) + '\n'); log.flush()
def unit(kind, arm, ids):
    fs = SQ if arm in 'QG' else SF; v = VRAM(); v.start(); r = {'unit': kind}
    if kind == 'both_deep':
        r['n_restored'] = [restore(0, fs['s0']), restore(1, fs['s1'])]; r['res'] = par([(ids[0:98000], 48, 0), (ids[200000:290000], 48, 1)])
        r['valid'] = all(r['res'][k]['http'] == 200 and (r['res'][k]['prompt_n'] or 1e9) < 64 and r['res'][k]['predicted_n'] == 48 for k in ('s0', 's1'))
    elif kind == 'single_98k':
        r['n_restored'] = [restore(0, fs['s0'])]; call('/slots/1?action=erase', {}); r['res'] = par([(ids[0:98000], 48, 0)])
        r['valid'] = r['res']['s0']['http'] == 200 and (r['res']['s0']['prompt_n'] or 1e9) < 64 and r['res']['s0']['predicted_n'] == 48
    else:   # append_4k
        r['n_restored'] = [restore(1, fs['s1'])]; call('/slots/0?action=erase', {}); r['res'] = par([(ids[200000:294096], 1, 1)])
        r['valid'] = r['res']['s1']['http'] == 200 and 4090 <= (r['res']['s1']['prompt_n'] or 0) <= 4200
    v.stop = True; r['vram_peak_mib'] = v.peak; return r
def main():
    out, order = sys.argv[1], sys.argv[2].split(','); smoke = '--smoke' in sys.argv; os.makedirs(out, exist_ok=True); os.makedirs(QSLOTS, exist_ok=True)
    seq = [(ci, pi, a) for ci, c in enumerate(order) for pi, a in enumerate(c)]; groups = []
    for x in seq:
        if groups and groups[-1][0] == x[2]: groups[-1][1].append(x)
        else: groups.append((x[2], [x]))
    log = open(out + '/units.jsonl', 'a')
    for arm, units in groups:
        for _ in range(180):
            if not gpu_apps(): break
            time.sleep(1)
        if gpu_apps(): sys.exit('GPU has compute apps before launch')
        name = 'nq8n-srv-%s-%d' % (arm.lower(), int(time.time()) % 100000); meta = launch(arm, name); sv = VRAM(); sv.start()
        try:
            t0 = time.time()
            while time.time() - t0 < 600:
                try:
                    if call('/health', method='GET', timeout=3)[0] == 200: break
                except Exception: pass
                time.sleep(2)
            else: raise RuntimeError('not ready')
            lg = sh(['docker', 'logs', name]); lg = lg.stdout + lg.stderr
            if arm == 'Q': assert 'occamy FA kvq8 on' in lg, 'native q8 path not enabled'
            else: assert 'occamy FA kvq8 on' not in lg
            ids = ids_load(); assert len(ids) >= 294096, len(ids)
            call('/completion', {'prompt': ids[:8], 'n_predict': 8, 'temperature': 0, 'id_slot': 0})   # warm
            if arm in 'QG' and not all(os.path.exists(QSLOTS + '/' + f) for f in SQ.values()): build_q8_slots(ids, log)
            for ci, pi, _ in units:
                for kind in ('both_deep', 'single_98k', 'append_4k'):
                    r = unit(kind, arm, ids); r.update(meta, cycle=ci, pos=pi, container=name, utc=now()); log.write(json.dumps(r) + '\n'); log.flush()
                    if not r['valid']: print('INVALID', arm, ci, pi, kind, json.dumps(r['res']), flush=True)
                if smoke: break
        finally:
            sv.stop = True; log.write(json.dumps({'session': name, 'arm': arm, 'session_vram_peak_mib': sv.peak, 'utc': now()}) + '\n'); log.flush()
            lg = sh(['docker', 'logs', name]); open(out + '/server-%s.log' % name, 'w').write((lg.stdout + lg.stderr)[-200000:]); sh(['docker', 'rm', '-f', name])
        if smoke: break
if __name__ == '__main__': main()
