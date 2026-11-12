#!/usr/bin/env python3
"""FGDIV arm: teacher-forced fg KL, cap0 vs cap1024, under real two-slot co-batch (bg stream + fg arrival), G1b build. Run only via run_all.sh (GPU lock held).
usage: g1c_fgdiv.py <outdir> <order e.g. AB,BA>   env G1C_GPU
Workload a (the only one with measured cross-arm divergence, identity pass idx 26): slot0 bg = 90K restored + 512 gen; slot1 fg = same 90K restored + 4096 append, 32 tokens.
fg ids_ref = G1b timed cap0 stream. Row 0 = fg arrival prefill (co-batched with bg decode); rows 1..31 = prompt+ids_ref[:j] one new token each (bg still decoding).
Arrival modes: T = after bg emitted 8 tokens (timed pattern), I = fixed 4 s after bg start (identity pattern, where cap1024 diverged). reps: T0, T1 (repeat noise), I0. Output <out>/<arm>_<mode>_<rep>.npy float32 logprobs [32, n_vocab]."""
import sys, os, json, time
sys.dont_write_bytecode = True
os.environ['G1B_GPU'] = os.environ['G1C_GPU']
sys.path.insert(0, '<LAB>/occamy35-day-20261112/evidence/G1b')
import g1b_run as g; import numpy as np
g.PORT = int(os.environ["G1C_PORT"]); CAP = {'A': 0, 'B': 1024}; G1C = os.path.dirname(os.path.abspath(__file__)); NVOC = 300000
def ids_ref():
    for l in open(g.G + '/run-20261112T191638Z/units.jsonl'):
        u = json.loads(l)
        if u['kind'] == 'a' and not u['identity_pass'] and u['arm'] == 'A': return u['fg']['ids']
def row(prompt, slot, nprobs):
    st, j = g.call('/completion', {'prompt': prompt, 'n_predict': 1, 'id_slot': slot, 'cache_prompt': True, 'return_tokens': True, 'temperature': 0, 'seed': 42, 'repeat_penalty': 1.0, 'n_probs': nprobs, 'post_sampling_probs': False})
    assert st == 200, (st, j); p = j['completion_probabilities'][0]['top_logprobs']
    v = np.full(max(e['id'] for e in p) + 1, -1e30, np.float32)
    for e in p: v[e['id']] = e['logprob']
    return v, j.get('timings', {})
def rep(arm, mode, r, tok90, ref, out):
    for sl in (0, 1):
        st, j = g.call('/slots/%d?action=restore' % sl, {'filename': 's1_90k_off200000.bin'}); assert st == 200 and j.get('n_restored') == 90000, (sl, st, j)
    bg = g.Stream(tok90[200000:290000] + [198], 0, 1536, True); bg.start()
    if mode == 'I': time.sleep(4)
    else: bg.ready.wait(120); assert len(bg.ev) >= 8 and bg.is_alive()
    fgp = tok90[200000:294096]; rows, tm = [], []
    v, t = row(fgp, 1, NVOC); rows.append(v); tm.append(t.get('prompt_n'))
    for j in range(1, 32):
        v, t = row(fgp + ref[:j], 1, NVOC); rows.append(v)
    alive = bg.is_alive(); bg.join(300)
    n = max(len(x) for x in rows); a = np.full((32, n), -1e30, np.float32)
    for i, x in enumerate(rows): a[i, :len(x)] = x
    np.save('%s/%s_%s_%d.npy' % (out, arm, mode, r), a); return dict(arm=arm, mode=mode, rep=r, fg_prompt_n_row0=tm[0], bg_alive_at_end=alive, nbg=len(bg.ev))
def main():
    out, order = sys.argv[1], sys.argv[2].split(','); os.makedirs(out, exist_ok=True)
    tok90, _ = g.load_tokens(); ref = ids_ref(); assert len(ref) == 32
    log = open(out + '/fgdiv_units.jsonl', 'a')
    for c in order:
        for arm in c:
            for _ in range(180):
                if not g.gpu_apps(): break
                time.sleep(1)
            if g.gpu_apps(): sys.exit('compute apps present before launch')
            name = 'fgdiv-%s-%d' % (arm.lower(), int(time.time()) % 100000)
            rc = g.sh([G1C + '/launch_fgdiv.sh', str(CAP[arm]), name[6:]]); assert rc.returncode == 0, rc.stderr
            try:
                t0 = time.time()
                while time.time() - t0 < 600:
                    try:
                        if g.call('/health', method='GET', timeout=3)[0] == 200: break
                    except Exception: pass
                    time.sleep(2)
                else: raise RuntimeError('not ready')
                g.call('/completion', {'prompt': [1, 2, 3, 4, 5, 6, 7, 8], 'n_predict': 8, 'temperature': 0, 'id_slot': 0})
                for mode, r in (('T', 0), ('T', 1), ('I', 0)):
                    if True: log.write(json.dumps(rep(arm, mode, r, tok90, ref, out)) + '\n'); log.flush()
            finally:
                open('%s/server_%s_%s.log' % (out, arm, name), 'w').write((lambda r: r.stdout + r.stderr)(g.sh(['docker', 'logs', 'g1c-' + name])))
                g.sh(['docker', 'rm', '-f', 'g1c-' + name])
if __name__ == '__main__': main()
