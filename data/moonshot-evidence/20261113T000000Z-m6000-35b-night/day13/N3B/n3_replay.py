#!/usr/bin/env python3
"""N3 two-slot agentic replay driver. Run ONLY via run_n3.sh (holds GPU lock, unloads, checks zero apps). CPU-only --plan is safe anywhere.
usage: n3_replay.py <outdir> [--plan] [--smoke] [--slots-only] [--no-sweep] [--no-ctxfull]
env: N3_GPU (device id/uuid, required unless --plan), N3_PROFILE=full|short, N3_START, N3_TURNS, N3_OFFSET_S, N3_THINK_S, N3_CUT_EPOCH,
     N3_BLOCK_ORDERS (e.g. "b3,gdn4,cap1024,gemm;gdn4,..." ; 5 blocks), N3_SWEEP_ORDER (e.g. "1024,512,2048,2048,512,1024"), N3_BIN_<ARM>=image|g1b
Layout: <outdir>/runs/<tag>/{run.json,telemetry.csv,server.log}; slots persist in <N3>/slots (built once by prefill with b3, saved with slot-save)."""
import sys, os, re, json, time, subprocess, threading, http.client, hashlib
M = '<LAB>'; E = M + '/occamy35-day-20261112/evidence'; N3 = M + '/occamy35-night-20261113/N3B'; SLOTS_RO = M + '/occamy35-night-20261113/N3/slots'   # N3B: slots read-only from N3 (evidence)
G1B = E + '/G1b/build/bin'; TOK = E + '/fixtures/tokens'; PORT = 18583
IMG = 'sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'
B3SH = M + '/run-35b-bundle3-20261112.sh'; GDN4SH = M + '/run-35b-gdn4-20261112.sh'
LIB = M + '/prod-libs/libggml-cuda-%s-20261112.so'
V4LIB = os.environ.get('N3B_GEMM_LIB', M + '/codex-investigation-20261112/prefill/gemm/integration-v4/src/build/bin/libggml-cuda.so')
ARMS = {  # arm: (lib, env-source script, cap, extra env, default binary: image=/app/llama-server of prod image; g1b=G1b build (cap patch))
    'gdn4': (LIB % 'gdn4', GDN4SH, 0, [], 'image'),
    'b3': (LIB % 'bundle3', B3SH, 0, [], 'g1b'),
    'cap1024': (LIB % 'bundle3', B3SH, 1024, [], 'g1b'),
    'gemm': (V4LIB, B3SH, 0, ['GGML_CUDA_FA_MAXWELL_GEMM=1'], 'g1b')}
WIL = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]; NAMES = ['b3', 'gdn4', 'cap1024', 'gemm']
ORDERS = [[NAMES[i] for i in r] for r in WIL] + [[NAMES[i] for i in WIL[0][::-1]]]   # 4 Williams rows (carryover balanced) + reversed row 0
PROFILES = {'full': dict(start=78000, turns=6), 'short': dict(start=84000, turns=4)}
P = PROFILES[os.environ.get('N3_PROFILE', 'full')]
START = int(os.environ.get('N3_START', P['start'])); TURNS = int(os.environ.get('N3_TURNS', P['turns']))
OFFSET = float(os.environ.get('N3_OFFSET_S', 25)); THINK = float(os.environ.get('N3_THINK_S', 1.0))
CUT = float(os.environ.get('N3_CUT_EPOCH', 0) or 0); SESS_EST = 660   # do not launch a session later than CUT - SESS_EST
SIZES = {'1K': (1024, 256), '4K': (4096, 512)}; TOOL_OFF = 120000; NCTX_SEQ = 98304
def kind_of(k): return '1K' if k % 2 == 0 else '4K'
def now(): return time.time()
def sh(c, **k): return subprocess.run(c, capture_output=True, text=True, **k)
def gpu(): return os.environ['N3_GPU']
def gpu_apps(): return sh(['nvidia-smi', '-i', gpu(), '--query-compute-apps=pid', '--format=csv,noheader']).stdout.strip()
def call(path, body=None, method='POST', timeout=3600):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'}); r = c.getresponse(); d = r.read(); c.close()
    try: return r.status, json.loads(d)
    except Exception: return r.status, {'raw': d[:400].decode(errors='replace')}
def prompt_sha(ids): return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
class Telemetry(threading.Thread):
    """1 Hz nvidia-smi sampler -> csv (epoch,sm_mhz,mem_mhz,power_w,mem_mib,temp_c); stop() at end of session."""
    Q = 'clocks.sm,clocks.mem,power.draw,memory.used,temperature.gpu'
    def __init__(s, path): super().__init__(daemon=True); s.f = open(path, 'w'); s.f.write('epoch,sm_mhz,mem_mhz,power_w,mem_mib,temp_c\n'); s.halt = False
    def run(s):
        while not s.halt:
            t = now(); o = sh(['nvidia-smi', '-i', gpu(), '--query-gpu=' + s.Q, '--format=csv,noheader,nounits']).stdout.strip()
            if o: s.f.write('%.2f,%s\n' % (t, o.replace(' ', ''))); s.f.flush()
            time.sleep(max(0.0, 1.0 - (now() - t)))
    def stop(s): s.halt = True; s.join(5); s.f.close()
def env_of(script):
    e = re.findall(r'-e (GGML_[A-Z0-9_]*=\S*)', open(script).read())
    assert not any('MAXWELL_GRAPHS' in x for x in e), 'MAXWELL_GRAPHS forbidden'; return e
def launch_cmd(arm, name, ub):
    lib, script, cap, extra, dbin = ARMS[arm]; binsel = os.environ.get('N3_BIN_' + arm.upper(), dbin)
    envs = env_of(script) + extra
    if arm == 'gemm':   # N3B preflight: refuse a lib without GEMM code
        n = int(sh('strings %s | grep -c FA_MAXWELL_GEMM' % lib.replace(' ', '\\ ') , shell=True).stdout.strip() or 0)
        if n == 0: sys.exit('PREFLIGHT FAIL: gemm lib has no FA_MAXWELL_GEMM: ' + lib)
        if not os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(lib))))) == 'integration-v4' and 'N3B_GEMM_LIB' not in os.environ: sys.exit('unexpected gemm lib path')
        if hashlib.sha256(open(lib, 'rb').read()).hexdigest()[:8] != '0c7ffbde': sys.exit('PREFLIGHT FAIL: gemm lib sha != 0c7ffbde')
    assert not any('MAXWELL_GRAPHS' in x for x in envs) and not os.environ.get('GGML_CUDA_MAXWELL_GRAPHS')
    if arm != 'gemm': assert not any('MAXWELL_GEMM' in x for x in envs) and not os.environ.get('GGML_CUDA_FA_MAXWELL_GEMM'), 'GEMM only in gemm arm'
    c = ['docker', 'run', '-d', '--name', name, '-p', '127.0.0.1:%d:8080' % PORT, '--device', 'nvidia.com/gpu=' + gpu(),
         '-v', '<MODELS>:/models:ro', '-v', SLOTS_RO + ':/slots:ro']
    if binsel == 'g1b': c += ['-v', G1B + ':/candidate:ro', '-v', lib + ':/candidate/libggml-cuda.so:ro', '-e', 'LD_LIBRARY_PATH=/candidate', '--entrypoint', '/candidate/llama-server']
    else: c += ['-v', lib + ':/app/libggml-cuda.so:ro']
    for x in envs: c += ['-e', x]
    c += [IMG, '-m', '/models/occamy-1.0-Q4_0-headq40b.gguf', '--alias', 'qwen36-moe', '--override-kv', 'tokenizer.ggml.pre=str:qwen35', '-ngl', '999', '-np', '2',
          '-c', '196608', '-b', '2048', '-ub', str(ub), '--cache-ram', '0', '--ctx-checkpoints', '16', '--checkpoint-min-step', '2048', '-fa', 'on',
          '--host', '0.0.0.0', '--port', '8080', '--jinja', '--metrics', '--slot-save-path', '/slots', '-lv', '4']
    if binsel == 'g1b': c += ['--mixed-batch-size', str(cap)]
    else: assert cap == 0, 'cap needs g1b binary'
    prof = os.environ.get('N3B_PROF_DIR')   # N3B: run the server under nvprof (trace csv in prof dir); image swapped to CUDA 12.8 devel (has nvprof)
    if prof:
        i = c.index(IMG); c[i:i + 1] = ['nvidia/cuda:12.8.1-devel-ubuntu24.04', '--print-gpu-trace', '--csv', '--log-file', '/out/trace.csv', '/candidate/llama-server']
        c[c.index('--entrypoint') + 1] = '/usr/local/cuda-12.8/bin/nvprof'; j = c.index('LD_LIBRARY_PATH=/candidate'); c[j] = 'LD_LIBRARY_PATH=/candidate:/usr/local/cuda-12.8/targets/x86_64-linux/lib'
        c[2:2] = ['--init', '-v', prof + ':/out']
        assert binsel == 'g1b'
    return c, {'arm': arm, 'binary': binsel, 'lib': os.path.basename(lib), 'env': envs, 'cap': cap, 'ub': ub}
class Req(threading.Thread):
    def __init__(s, ids, slot, n, tag):
        super().__init__(daemon=True); s.ids, s.slot, s.n, s.tag = ids, slot, n, tag; s.ev = []; s.final = None; s.err = None; s.status = None; s.body = None
        s.t_send = s.t_end = None
    def run(s):
        try:
            c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
            b = {'prompt': s.ids, 'n_predict': s.n, 'id_slot': s.slot, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True, 'stream': True, 'return_tokens': True}
            s.t_send = now(); c.request('POST', '/completion', json.dumps(b), {'Content-Type': 'application/json'}); r = c.getresponse(); s.status = r.status
            if r.status != 200: s.body = r.read()[:600].decode(errors='replace'); return
            for line in r:
                if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
                j = json.loads(line[5:]); t = now()
                if j.get('stop'): s.final = j
                else: s.ev.append((t, j.get('content', ''), j.get('tokens', [])))
        except Exception as ex: s.err = repr(ex)
        finally: s.t_end = now()
    def rec(s):
        f = s.final or {}; tm = f.get('timings', {}); ids = [i for e in s.ev for i in e[2]]; npred = f.get('tokens_predicted')
        nonascii = sum(1 for e in s.ev if e[2] and any(ord(ch) > 127 for ch in e[1]))
        r = {'tag': s.tag, 'slot': s.slot, 'prompt_len': len(s.ids), 'n_predict': s.n, 't_send': s.t_send, 't_end': s.t_end, 't_first': s.ev[0][0] if s.ev else None,
             'ev': [[e[0], len(e[2])] for e in s.ev], 'tokens_predicted': npred, 'n_ids_visible': len(ids), 'id_gap': (npred - len(ids)) if npred is not None else None,
             'nonascii_events': nonascii, 'http': s.status, 'body': s.body, 'err': s.err, 'prompt_sha': prompt_sha(s.ids),
             'content_sha': hashlib.sha256(''.join(e[1] for e in s.ev).encode()).hexdigest()[:16], 'prompt_n': tm.get('prompt_n'), 'cache_n': tm.get('cache_n'),
             'prompt_ms': tm.get('prompt_ms'), 'predicted_ms': tm.get('predicted_ms'), 'prompt_tps': tm.get('prompt_per_second'), 'predicted_tps': tm.get('predicted_per_second'),
             'tokens_evaluated': f.get('tokens_evaluated'), 'tokens_cached': f.get('tokens_cached'), 'truncated': f.get('truncated'), 'stop_type': f.get('stop_type')}
        r['receipt_ok'] = bool(npred == s.n and not f.get('truncated') and r['id_gap'] is not None and 0 <= r['id_gap'] <= nonascii)
        return r, ids
def corpus(a): return json.load(open('%s/d%d.json' % (TOK, a)))
def tool_chunks(a, turns):
    c = corpus(a); pos = TOOL_OFF; out = []
    for k in range(turns):
        n = SIZES[kind_of(k)][0]; out.append(c[pos:pos + n]); pos += n
    assert all(len(x) == SIZES[kind_of(k)][0] for k, x in enumerate(out)), 'corpus too short'; return out
def start_container(arm, ub, tag, outd):
    for _ in range(180):   # lingering compute context after previous docker rm
        if not gpu_apps(): break
        time.sleep(1)
    if gpu_apps(): sys.exit('GPU has compute apps before launch: ' + gpu_apps())
    name = 'n3-%s-%d' % (tag.replace('_', '-')[:24], int(time.time()) % 100000); cmd, meta = launch_cmd(arm, name, ub)
    rc = sh(cmd)
    if rc.returncode: raise RuntimeError('docker run failed: ' + rc.stderr[-300:])
    t0 = now()
    while now() - t0 < 600:
        if sh(['docker', 'inspect', '-f', '{{.State.Running}}', name]).stdout.strip() != 'true': raise RuntimeError('container exited (OOM/launch fail)')
        try:
            if call('/health', method='GET', timeout=3)[0] == 200: break
        except Exception: pass
        time.sleep(2)
    else: raise RuntimeError('not ready 600 s')
    meta['container'] = name; meta['ready_s'] = now() - t0; return name, meta
def stop_container(name, outd):
    if os.environ.get('N3B_PROF_DIR'):
        sh(['docker', 'exec', name, 'sh', '-c', 'kill -INT $(pidof llama-server)'])
        for _ in range(90):
            if not sh(['docker', 'ps', '-q', '-f', 'name=' + name]).stdout.strip(): break
            time.sleep(2)
    r = sh(['docker', 'logs', name]); open(outd + '/server.log', 'w').write(r.stdout + r.stderr); sh(['docker', 'rm', '-f', name])
def slot_files(): return ['n3_s%d_%d.bin' % (a, START) for a in (0, 1)], '%s/n3_slots_%d.json' % (SLOTS_RO, START)
def build_slots(out):
    """Prefill corpus[:START] per slot once (arm b3), slot-save, verify restore count. Skipped if files+meta exist."""
    files, meta = slot_files()
    if os.path.exists(meta) and all(os.path.exists('%s/%s' % (SLOTS_RO, f)) for f in files): print('slots present, skip build', flush=True); return
    os.makedirs(N3 + '/slots', exist_ok=True); d = out + '/runs/slotbuild'; os.makedirs(d, exist_ok=True)
    tel = Telemetry(d + '/telemetry.csv'); tel.start(); name = None
    try:
        name, lm = start_container('b3', 1024, 'slotbuild', d); rs = []
        rq = [Req(corpus(a)[:START], a, 1, 'build%d' % a) for a in (0, 1)]; t0 = now()
        for r in rq: r.start()
        for r in rq: r.join(5400)
        info = {'start': START, 'launch': lm, 'build_s': now() - t0, 'slots': []}
        for a, r in enumerate(rq):
            rec, ids = r.rec(); assert r.status == 200 and rec['tokens_predicted'] == 1, ('build failed', rec['http'], rec['err'], rec['body'])
            st, j = call('/slots/%d?action=save' % a, {'filename': files[a]}); assert st == 200, (st, j)
            info['slots'].append({'save': j, 'gen_ids': ids, 'prompt_n': rec['prompt_n']})
        for a in (0, 1):   # verify restore count; prefix = corpus[:START] (+ the 1 sampled token if the slot file holds it)
            call('/slots/%d?action=erase' % a, {}); st, j = call('/slots/%d?action=restore' % a, {'filename': files[a]}); n = j.get('n_restored')
            ex = info['slots'][a]['gen_ids'] if n == START + len(info['slots'][a]['gen_ids']) else []
            assert n == START or ex, ('restore count mismatch', a, n); info['slots'][a].update(n_restored=n, prefix_extra=ex)
        json.dump(info, open(meta, 'w'))
    finally:
        tel.stop()
        if name: stop_container(name, d)
def prefix_for(a):
    files, meta = slot_files(); m = json.load(open(meta)); return corpus(a)[:START] + m['slots'][a]['prefix_extra']
def agent(a, turns, t_start, out, ids, tools, stop_ev):
    while now() < t_start: time.sleep(0.01)
    for k in range(turns):
        ids = ids + ([198] if k else []) + tools[k]; q = Req(list(ids), a, SIZES[kind_of(k)][1], 'a%d_t%d_%s' % (a, k, kind_of(k))); q.start(); q.join(1800)
        rec, gen = q.rec(); rec.update(agent=a, turn=k, kind=kind_of(k), arrival_sched=t_start); out.append(rec)
        if q.status != 200 or q.err: stop_ev.append(a); return
        ids = ids + gen; time.sleep(THINK)
    out.append({'tag': 'agent_final', 'agent': a, 'final_len': len(ids), '_ids': ids})
def run_session(out, arm, ub, tag, blk, kind, turns, ctxfull):
    d = '%s/runs/%s' % (out, tag); os.makedirs(d, exist_ok=True); files, _ = slot_files()
    tel = Telemetry(d + '/telemetry.csv'); tel.start(); name = None; R = {'tag': tag, 'arm': arm, 'ub': ub, 'block': blk, 'kind': kind, 'turns': turns, 'start': START,
                                                                       'offset_s': OFFSET, 'think_s': THINK, 'utc_start': now(), 'ok': False}
    try:
        name, lm = start_container(arm, ub, tag, d); R['launch'] = lm
        call('/completion', {'prompt': [1, 2, 3, 4, 5, 6, 7, 8], 'n_predict': 4, 'temperature': 0, 'id_slot': 0})   # warm
        R['restore'] = []
        for a in (0, 1):
            t = now(); st, j = call('/slots/%d?action=restore' % a, {'filename': files[a]}); R['restore'].append({'slot': a, 'http': st, 's': now() - t, 'n_restored': j.get('n_restored')}); assert st == 200, (st, j)
        res = {0: [], 1: []}; stop = []; t0 = now() + 2.0; R['t_run0'] = t0
        th = [threading.Thread(target=agent, args=(a, turns, t0 + (0 if a == 0 else OFFSET), res[a], prefix_for(a), tool_chunks(a, turns), stop)) for a in (0, 1)]
        for t in th: t.start()
        for t in th: t.join(3600)
        R['reqs'] = [r for a in (0, 1) for r in res[a] if r['tag'] != 'agent_final']; fin = {r['agent']: r for a in (0, 1) for r in res[a] if r['tag'] == 'agent_final'}
        R['final_len'] = {a: fin[a]['final_len'] for a in fin}; R['ok'] = len(R['reqs']) == 2 * turns and all(r['receipt_ok'] for r in R['reqs'])
        if ctxfull:   # N3b: push slot 0 past n_ctx_seq with an extra tool result; record server behaviour + cost, then probe health and the other slot
            ids0 = fin[0]['_ids']; need = max(4096, NCTX_SEQ - len(ids0) + 1500); big = ids0 + [198] + corpus(0)[TOOL_OFF + 20000:TOOL_OFF + 20000 + need]
            q = Req(big, 0, 64, 'ctxfull'); q.start(); q.join(1800); rec, _ = q.rec()
            try: hs = call('/health', method='GET', timeout=10)[0]
            except Exception as ex: hs = repr(ex)
            q1 = Req(fin[1]['_ids'] + [198] + corpus(1)[TOOL_OFF + 20000:TOOL_OFF + 20256], 1, 16, 'ctxfull_probe_slot1'); q1.start(); q1.join(600); rec1, _ = q1.rec()
            R['ctxfull'] = {'prompt_len': len(big), 'n_ctx_seq': NCTX_SEQ, 'req': rec, 'health_after': hs, 'probe_slot1': rec1, 'wall_s': (rec['t_end'] or now()) - rec['t_send']}
    except Exception as ex: R['error'] = repr(ex); print('SESSION FAIL', tag, repr(ex), flush=True)
    finally:
        tel.stop(); R['utc_end'] = now()
        if name: stop_container(name, d)
        json.dump(R, open(d + '/run.json', 'w'))
    return R
def cut_ok():
    return not CUT or now() + SESS_EST < CUT
def plan():
    per = sum(SIZES[kind_of(k)][0] + SIZES[kind_of(k)][1] for k in range(TURNS)) + (TURNS - 1)
    print('profile=%s start=%d turns=%d append+gen/slot=%d final~%d (limit 96000; ctx %d) offset=%ss think=%ss' % (os.environ.get('N3_PROFILE', 'full'), START, TURNS, per, START + per, NCTX_SEQ, OFFSET, THINK))
    for b, o in enumerate(block_orders()): print('block %d: %s%s' % (b, ' > '.join(o), '  (+ctxfull each arm)' if b == len(block_orders()) - 1 else ''))
    print('sweep (b3 only, 2 turns): ub order', sweep_order(), '| arm bins:', {a: os.environ.get('N3_BIN_' + a.upper(), v[4]) for a, v in ARMS.items()})
    if os.environ.get('N3_GPU'):
        for a in ARMS: l = launch_cmd(a, 'n3-x', 1024)[1]; print(a, 'bin', l['binary'], 'lib', l['lib'], 'cap', l['cap'], 'n_env', len(l['env']))
def block_orders():
    o = os.environ.get('N3_BLOCK_ORDERS'); return [x.split(',') for x in o.split(';')] if o else ORDERS
def sweep_order(): return [int(x) for x in os.environ.get('N3_SWEEP_ORDER', '1024,512,2048,2048,512,1024').split(',')]
def main():
    a = [x for x in sys.argv[1:] if not x.startswith('--')]; fl = {x for x in sys.argv[1:] if x.startswith('--')}
    if '--plan' in fl: plan(); return
    out = a[0]; os.makedirs(out + '/runs', exist_ok=True); json.dump({'start': START, 'turns': TURNS, 'orders': block_orders(), 'sweep': sweep_order(), 'offset_s': OFFSET}, open(out + '/plan.json', 'w'))
    build_slots(out)
    if '--slots-only' in fl: return
    if '--prof' in fl:
        pd = out + '/prof'; os.makedirs(pd, exist_ok=True); os.environ['N3B_PROF_DIR'] = pd
        run_session(out, 'gemm', 1024, 'prof_gemm', 0, 'prof', min(TURNS, 2), False); del os.environ['N3B_PROF_DIR']
        if '--prof-only' in fl: return
    elif '--smoke' in fl: run_session(out, 'b3', 1024, 'smoke_b3', 0, 'smoke', min(TURNS, 2), False); return
    orders = block_orders()
    for b, order in enumerate(orders):
        for pos, arm in enumerate(order):
            if not cut_ok(): print('CUT reached; stop before block', b, arm, flush=True); return
            run_session(out, arm, 1024, 'b%d_p%d_%s' % (b, pos, arm), b, 'main', TURNS, ctxfull=(b == len(orders) - 1 and '--no-ctxfull' not in fl))
    if '--no-sweep' in fl: return
    for i, ub in enumerate(sweep_order()):
        if not cut_ok(): print('CUT reached; stop sweep', flush=True); return
        run_session(out, 'b3', ub, 'sw%d_ub%d' % (i, ub), i, 'sweep', min(TURNS, 2), False)
if __name__ == '__main__': main()
