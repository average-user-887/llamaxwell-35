#!/usr/bin/env python3
"""S0 soak driver. Run ONLY via s0_gpu.sh (GPU lock held, unload done, zero compute apps). Reuses the L0 prod-arm launch (n3_replay_l0.launch_cmd) with an
own container prefix ns0- and a writable slot dir. usage: soak.py <outdir>.   env: N3_GPU, SOAK_MIN (default 60), N3_* as set by s0_gpu.sh.
Per slot: episodes of [prefill ~84K base, 4 tool-append turns (1K/4K, like N3 short) -> ~96K]. Episode e: restore/identity check after turn 1 (every episode),
mid-prefill cancel before turn 0 on odd episodes. Save/restore uses the server slot API (--slot-save-path); fallback if save is refused: evict slot with another prompt and replay.
Files: turns.jsonl (one event per line), vram.csv (5 s), soak.json (meta), errors.log, server.log (from stop_container)."""
import sys, os, json, time, threading, http.client, importlib.util, random, hashlib
HERE = '<W>'
spec = importlib.util.spec_from_file_location('l0', HERE + '/L0/n3_replay_l0.py'); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
OUT = sys.argv[1]; SOAK_S = float(os.environ.get('SOAK_MIN', 60)) * 60; BASE_LEN = 84000; NTURN = 4; GEN_CHK = 64
now = L.now; call = L.call; Req = L.Req
SECRET = [os.environ.get(k) for k in ('N3_GPU', 'N3_UNLOAD_URL') if os.environ.get(k)]
def scrub(s):
    s = str(s)
    for x in SECRET: s = s.replace(x, '<redacted>')
    return s
LK = threading.Lock()
def ev(**k):
    k['t'] = now()
    with LK: open(OUT + '/turns.jsonl', 'a').write(scrub(json.dumps(k)) + '\n')
def err(msg):
    with LK: open(OUT + '/errors.log', 'a').write('%.1f %s\n' % (now(), scrub(msg)))
class Gate:
    """req(): ordinary request in flight. exclusive(): run with every other request drained (identity check needs a quiet server)."""
    def __init__(s): s.c = threading.Condition(); s.n = 0; s.excl = False
    def req(s):
        g = s
        class C:
            def __enter__(c):
                with g.c:
                    while g.excl: g.c.wait()
                    g.n += 1
            def __exit__(c, *a):
                with g.c: g.n -= 1; g.c.notify_all()
        return C()
    def exclusive(s):
        g = s
        class C:
            def __enter__(c):
                with g.c:
                    while g.excl: g.c.wait()
                    g.excl = True
                    while g.n: g.c.wait()
            def __exit__(c, *a):
                with g.c: g.excl = False; g.c.notify_all()
        return C()
gate = Gate(); abort = threading.Event(); T0 = None; DEADLINE = None
class Vram(threading.Thread):
    def __init__(s, path): super().__init__(daemon=True); s.f = open(path, 'w'); s.f.write('epoch,used_mib,total_mib,free_mib,power_w,temp_c\n'); s.halt = False
    def run(s):
        while not s.halt:
            t = now(); o = L.sh(['nvidia-smi', '-i', L.gpu(), '--query-gpu=memory.used,memory.total,power.draw,temperature.gpu', '--format=csv,noheader,nounits']).stdout.strip()
            try:
                u, tot, p, tc = [x.strip() for x in o.split(',')]; s.f.write('%.1f,%s,%s,%d,%s,%s\n' % (t, u, tot, int(float(tot)) - int(float(u)), p, tc)); s.f.flush()
            except Exception: pass
            time.sleep(max(0.0, 5.0 - (now() - t)))
    def stop(s): s.halt = True; s.join(8); s.f.close()
def alive(name): return L.sh(['docker', 'inspect', '-f', '{{.State.Running}}', name]).stdout.strip() == 'true'
def monitor(name):
    while not abort.is_set() and now() < DEADLINE + 3600:
        time.sleep(15)
        if not alive(name): err('container died'); ev(kind='container_died'); abort.set(); return
def do_req(ids, slot, n, tag, **extra):
    with gate.req(): return one(ids, slot, n, tag, **extra)
def one(ids, slot, n, tag, **extra):
    q = Req(list(ids), slot, n, tag); q.start(); q.join(2400); rec, gen = q.rec()
    if q.is_alive(): rec['err'] = 'join timeout'
    bad = rec['http'] != 200 or rec['err'] or not rec['receipt_ok']
    rec.update(kind='turn', slot=slot, bad=bool(bad), **extra); ev(**{k: v for k, v in rec.items() if k != 'ev'})
    if bad: err('bad request %s http=%s err=%s body=%s receipt_ok=%s' % (tag, rec['http'], rec['err'], rec['body'], rec['receipt_ok']))
    return rec, gen, bool(bad)
def slot_idle(slot):
    try:
        st, j = call('/slots', method='GET', timeout=10)
        if st == 200 and isinstance(j, list): return not any(s.get('is_processing') for s in j if s.get('id') == slot)
    except Exception: pass
    return None
def cancel_mid_prefill(ids, slot, tag, wait_s):
    """Send a streaming completion, never read a byte, close the socket wait_s later (prefill of a fresh ~84K prompt takes minutes). Then time until the slot is idle."""
    with gate.req():
        c = http.client.HTTPConnection('127.0.0.1', L.PORT, timeout=60)
        b = {'prompt': ids, 'n_predict': 64, 'id_slot': slot, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True, 'stream': True, 'return_tokens': True}
        t = now(); c.request('POST', '/completion', json.dumps(b), {'Content-Type': 'application/json'}); time.sleep(wait_s); c.close(); tc = now()
        idle = None; t1 = now()
        while now() - t1 < 300:
            r = slot_idle(slot)
            if r is None: break
            if r: idle = now() - tc; break
            time.sleep(1)
        try: hs = call('/health', method='GET', timeout=10)[0]
        except Exception as e: hs = repr(e)
    ev(kind='cancel', slot=slot, tag=tag, after_s=wait_s, idle_after_s=idle, honoured=(idle is not None and idle < 120), health=hs, prompt_len=len(ids))
    if idle is None or idle >= 120 or hs != 200: err('cancel %s idle_after_s=%s health=%s' % (tag, idle, hs))
def identity_check(ids, slot, tag, fn):
    """ids = full context now cached in the slot. pre = greedy 64 on live slot; save; erase; restore; post = greedy 64. Run with the server otherwise quiet."""
    with gate.exclusive():
        pr = ids + [198]; R = {'kind': 'restore_check', 'tag': tag, 'slot': slot, 'ctx_len': len(ids)}
        st, j = call('/slots/%d?action=save' % slot, {'filename': fn}); R['save_http'] = st; R['save'] = {k: j.get(k) for k in ('n_saved', 'n_written', 'save_ms')} if isinstance(j, dict) else None
        # save happens BEFORE pre so the file holds exactly `ids` (pre extends the live cache by 64 tokens)
        r1, g1, b1 = one(pr, slot, GEN_CHK, tag + '_pre', phase='chk_pre')
        if st == 200:
            R['mode'] = 'slot_api'; call('/slots/%d?action=erase' % slot, {}); t = now(); st2, j2 = call('/slots/%d?action=restore' % slot, {'filename': fn})
            R['restore_http'] = st2; R['restore_s'] = now() - t; R['n_restored'] = j2.get('n_restored') if isinstance(j2, dict) else None
        else:   # fallback: evict with another 90K-class prompt (other corpus), replay
            R['mode'] = 'evict_replay'; other = L.corpus(1 - slot)[60000:60000 + BASE_LEN]; one(other, slot, 1, tag + '_evict', phase='chk_evict'); st2 = 200
        r2, g2, b2 = one(pr, slot, GEN_CHK, tag + '_post', phase='chk_post')
        R.update(pre_sha=hashlib.sha256(json.dumps(g1).encode()).hexdigest()[:16], post_sha=hashlib.sha256(json.dumps(g2).encode()).hexdigest()[:16], n_pre=len(g1), n_post=len(g2),
                 post_cache_n=r2.get('cache_n'), post_prompt_n=r2.get('prompt_n'), identical=(g1 == g2 and len(g1) == GEN_CHK and not (b1 or b2) and st2 == 200),
                 first_diff=next((i for i, (a, b) in enumerate(zip(g1, g2)) if a != b), None))
        ev(**R)
        if not R['identical']: err('restore identity FAIL %s %s' % (tag, json.dumps({k: R[k] for k in ('mode', 'n_pre', 'n_post', 'first_diff', 'restore_http', 'n_restored') if k in R})))
        return g2
def agent(a, t_start):
    rnd = random.Random(1000 + a); c = L.corpus(a); tools = L.tool_chunks(a, NTURN); e = 0
    while now() < t_start and not abort.is_set(): time.sleep(0.05)
    while now() < DEADLINE and not abort.is_set():
        off = (e * 4000 + a * 2000) % 30000; base = c[off:off + BASE_LEN]; tag = 's%d_e%d' % (a, e); ev(kind='episode', slot=a, episode=e, off=off)
        if e % 2 == 1: cancel_mid_prefill(base + tools[0], a, tag + '_cancel', rnd.uniform(20, 50))
        ids = list(base); fail = False
        for k in range(NTURN):
            if now() >= DEADLINE or abort.is_set(): break
            ids = ids + ([198] if k else []) + tools[k]; n = L.SIZES[L.kind_of(k)][1]
            rec, gen, bad = do_req(ids, a, n, '%s_t%d_%s' % (tag, k, L.kind_of(k)), episode=e, turn=k, size=L.kind_of(k), first=(k == 0), phase='turn')
            if bad: fail = True; break
            ids = ids + gen
            if k == 1 and now() < DEADLINE:
                g2 = identity_check(ids, a, tag + '_chk', 's0_slot%d.bin' % a); ids = ids + [198] + g2   # continue the agent from the post-restore state
            time.sleep(L.THINK)
        e += 1
        if fail: time.sleep(10)
def main():
    global T0, DEADLINE
    os.makedirs(OUT, exist_ok=True); meta = {'soak_min': SOAK_S / 60, 'base_len': BASE_LEN, 'turns_per_episode': NTURN, 'gen_check': GEN_CHK, 'utc_start': now()}
    vr = Vram(OUT + '/vram.csv'); vr.start(); name = None
    try:
        for _ in range(180):
            if not L.gpu_apps(): break
            time.sleep(1)
        if L.gpu_apps(): sys.exit('GPU has compute apps before launch')
        name = 'ns0-soak-%d' % (int(time.time()) % 100000); cmd, lm = L.launch_cmd('prod', name, L.ub_of('prod', 1024))
        i = [x for x in cmd if x.endswith(':/slots:ro')]; assert len(i) == 1; cmd[cmd.index(i[0])] = OUT + '/slots:/slots'
        assert 'GGML_CUDA_MAXWELL_GRAPHS' not in ' '.join(cmd)
        rc = L.sh(cmd)
        if rc.returncode: raise RuntimeError('docker run failed: ' + scrub(rc.stderr[-300:]))
        t0 = now()
        while now() - t0 < 600:
            if not alive(name): raise RuntimeError('container exited at launch')
            try:
                if call('/health', method='GET', timeout=3)[0] == 200: break
            except Exception: pass
            time.sleep(2)
        else: raise RuntimeError('not ready 600 s')
        lm.update(container=name, ready_s=now() - t0); meta['launch'] = {k: lm[k] for k in ('arm', 'binary', 'lib', 'env', 'cap', 'ub')}
        call('/completion', {'prompt': [1, 2, 3, 4, 5, 6, 7, 8], 'n_predict': 4, 'temperature': 0, 'id_slot': 0})
        T0 = now() + 1; DEADLINE = T0 + SOAK_S; meta['t0'] = T0; meta['deadline'] = DEADLINE
        ths = [threading.Thread(target=agent, args=(a, T0 + (0 if a == 0 else L.OFFSET))) for a in (0, 1)] + [threading.Thread(target=monitor, args=(name,), daemon=True)]
        for t in ths: t.start()
        for t in ths[:2]: t.join(SOAK_S + 3600)
        abort.set()
    except SystemExit as ex: meta['error'] = scrub(ex); err('exit ' + scrub(ex))
    except Exception as ex: meta['error'] = scrub(repr(ex)); err('exception ' + scrub(repr(ex))); print('SOAK FAIL', scrub(repr(ex)), flush=True)
    finally:
        vr.stop(); meta['utc_end'] = now()
        if name:
            L.stop_container(name, OUT, graceful=True); meta['gemm_check'] = L.gemm_check(OUT + '/server.log')
        json.dump(meta, open(OUT + '/soak.json', 'w'), indent=1, default=str)
main()
