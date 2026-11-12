#!/usr/bin/env python3
"""G1b runner: cap0 vs cap1024, workloads a (restored-90K append) and b (deep stream + 16K arrival). Run ONLY via run-g1b.sh (holds GPU lock).
usage: g1b_run.py <outdir> <order e.g. ABBA,BAAB,ABBA,BAAB,ABBA> [--smoke]"""
import sys, os, json, time, subprocess, threading, http.client, hashlib, datetime
M = '<LAB>'; E = M + '/occamy35-day-20261112/evidence'; G = E + '/G1b'
CODEX = M + '/codex-investigation-20261112'; PORT = 18493; CAP = {'A': 0, 'B': 1024}
GPU = os.environ['G1B_GPU']
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
class Stream(threading.Thread):
    """streaming /completion with return_tokens. Records per-event (t, content, ids). stream=False -> exact IDs from final.tokens (identity pass)."""
    def __init__(s, prompt, slot, n, stream=True, gate=None):
        super().__init__(daemon=True); s.p, s.slot, s.n, s.stream, s.gate = prompt, slot, n, stream, gate
        s.ev = []; s.final = None; s.err = None; s.t_send = s.t_end = None; s.ready = threading.Event()
    def run(s):
        try:
            if s.gate: s.gate()
            c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=900)
            body = {'prompt': s.p, 'n_predict': s.n, 'id_slot': s.slot, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True, 'stream': s.stream, 'return_tokens': True}
            s.t_send = time.monotonic(); c.request('POST', '/completion', json.dumps(body), {'Content-Type': 'application/json'}); r = c.getresponse()
            if r.status != 200: s.err = 'HTTP %d' % r.status; return
            if not s.stream: s.final = json.loads(r.read()); return
            for line in r:
                if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
                j = json.loads(line[5:]); t = time.monotonic()
                if j.get('stop'): s.final = j
                else:
                    s.ev.append((t, j.get('content', ''), j.get('tokens', [])))
                    if len(s.ev) >= 8: s.ready.set()
        except Exception as ex: s.err = repr(ex)
        finally: s.t_end = time.monotonic(); s.ready.set()
def summarize(s):
    ids = [i for e in s.ev for i in e[2]]; f = s.final or {}; tm = f.get('timings', {})
    ts = [e[0] for e in s.ev if e[2]]; n_gen = f.get('tokens_predicted')
    nonascii_merge = sum(1 for e in s.ev if e[2] and any(ord(ch) > 127 for ch in e[1]))
    return {'err': s.err, 't_send': s.t_send, 't_end': s.t_end, 'ts': ts, 'n_events': len(s.ev), 'ids': ids if s.stream else f.get('tokens', []),
            'n_gen_server': n_gen, 'n_ids_visible': len(ids), 'id_gap': (n_gen - len(ids)) if (s.stream and n_gen is not None) else None, 'nonascii_events': nonascii_merge,
            'content_sha256': hashlib.sha256(''.join(e[1] for e in s.ev).encode() if s.stream else str(f.get('content', '')).encode()).hexdigest(),
            'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'), 'prompt_per_second': tm.get('prompt_per_second'), 'predicted_n': tm.get('predicted_n'),
            'tokens_evaluated': f.get('tokens_evaluated'), 'tokens_cached': f.get('tokens_cached'), 'truncated': f.get('truncated'), 'stop_type': f.get('stop_type')}
def gaps(ts, a, b):   # max gap between consecutive visible-token events overlapping [a,b]
    g = [(t1 - t0) for t0, t1 in zip(ts, ts[1:]) if t1 >= a and t0 <= b]; return max(g) if g else None
def load_tokens():
    return json.load(open(CODEX + '/fixture_tokens.json')), json.load(open(E + '/fixtures/tokens/d0.json'))
def unit(kind, tok90, d0, useg, ident):
    """kind a|b; ident=True -> stream:false exact-ID pass (arrival after fixed 4 s)."""
    v = VRAM(); v.start(); t_unit = time.monotonic()
    if kind == 'a':
        for sl in (0, 1):
            st, j = call('/slots/%d?action=restore' % sl, {'filename': 's1_90k_off200000.bin'}); assert st == 200 and j.get('n_restored') == 90000, (sl, st, j)
        bg = Stream(tok90[200000:290000] + [198], 0, 512, not ident); fg = Stream(tok90[200000:294096], 1, 32, not ident)
    else:
        st, j = call('/slots/0?action=restore', {'filename': 'd0.bin'}); assert st == 200, (st, j); call('/slots/1?action=erase', {}); time.sleep(1)
        u = json.load(open(E + '/fixtures/tokens/u%02d.json' % useg))[:16384]
        bg = Stream(d0[:98047] + [198], 0, 256, not ident); fg = Stream(u, 1, 32, not ident)
    bg.start()
    if ident: time.sleep(4)
    else: bg.ready.wait(120); assert len(bg.ev) >= 8 and bg.is_alive(), 'bg not decoding at arrival'
    t_arr = time.monotonic(); fg.start(); fg.join(900); bg.join(900); t1 = time.monotonic(); v.stop = True
    B, F = summarize(bg), summarize(fg); r = {'kind': kind, 'identity_pass': ident, 'bg': B, 'fg': F, 'vram_peak_mib': v.peak}
    if not ident and not (B['err'] or F['err']):
        first_fg = F['ts'][0]
        r.update(ttft_s=first_fg - F['t_send'], arrival_to_first_fg_s=first_fg - t_arr, wall_s=t1 - bg['t_send'] if False else t1 - bg.t_send,
                 max_gap_s=gaps(B['ts'], t_arr, first_fg), fg_prefill_tps=F['prompt_per_second'])
    r['valid'] = (not (B['err'] or F['err'])) and ((F['prompt_n'] or 0) >= (4090 if kind == 'a' else 16000)) and not F['truncated']
    return r
def main():
    out, order = sys.argv[1], sys.argv[2].split(','); smoke = '--smoke' in sys.argv; os.makedirs(out, exist_ok=True)
    tok90, d0 = load_tokens(); seq = [(ci, pi, arm) for ci, c in enumerate(order) for pi, arm in enumerate(c)]
    groups = []
    for x in seq:
        if groups and groups[-1][0] == x[2]: groups[-1][1].append(x)
        else: groups.append((x[2], [x]))
    seen_id = set(); log = open(out + '/units.jsonl', 'a')
    for arm, units in groups:
        for _ in range(180):   # previous g1b container's compute context can linger a few s after docker rm
            if not gpu_apps(): break
            time.sleep(1)
        if gpu_apps(): sys.exit('GPU has compute apps before launch: ' + gpu_apps())
        name = 'g1b-%s-%d' % (arm.lower(), int(time.time()) % 100000)
        rc = sh([G + '/launch-arm.sh', str(CAP[arm]), name[4:]]); assert rc.returncode == 0, rc.stderr
        try:
            t0 = time.time()
            while time.time() - t0 < 600:
                try:
                    if call('/health', method='GET', timeout=3)[0] == 200: break
                except Exception: pass
                time.sleep(2)
            else: raise RuntimeError('not ready')
            call('/completion', {'prompt': [1, 2, 3, 4, 5, 6, 7, 8], 'n_predict': 8, 'temperature': 0, 'id_slot': 0})   # warm
            open(out + '/server-%s-%s.log' % (arm, name), 'w').write((lambda r: r.stdout + r.stderr)(sh(['docker', 'logs', name]))[-20000:])
            for ci, pi, _ in units:
                for kind in ('a', 'b'):
                    passes = [False] + ([True] if (ci == 0 and (arm, kind) not in seen_id) else [])
                    for ident in passes:
                        r = unit(kind, tok90, d0, (ci * 4 + pi) % 16, ident); r.update(arm=arm, cap=CAP[arm], cycle=ci, pos=pi, container=name, utc=now())
                        log.write(json.dumps(r) + '\n'); log.flush()
                        if ident: seen_id.add((arm, kind))
                        if not r['valid']: print('INVALID unit', arm, ci, pi, kind, flush=True)
                if smoke: break
        finally:
            open(out + '/server-%s-%s.full.log' % (arm, name), 'w').write((lambda r: r.stdout + r.stderr)(sh(['docker', 'logs', name])))
            sh(['docker', 'rm', '-f', name])
        if smoke: break
if __name__ == '__main__': main()
