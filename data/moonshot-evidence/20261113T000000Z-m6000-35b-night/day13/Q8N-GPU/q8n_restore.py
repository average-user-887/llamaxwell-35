#!/usr/bin/env python3
"""Q8N slot-restore test (call only from run-q8n-gpu.sh, which holds the GPU lock). usage: q8n_restore.py <outdir>   env Q8N_GPU
Native q8 server. Builds q8 slot files if absent (prefill s0 98000 n=24, save; s1 90000 n=1, save); the first 16 tokens of the s0 build decode are the
no-restore continuation. Then restore s0, decode 16 tokens of the same prompt, compare token ids; prompt_n>=64 means the restore re-prefilled."""
import sys, os, json, time, importlib.util
spec = importlib.util.spec_from_file_location('q8n_srv', os.path.dirname(os.path.abspath(__file__)) + '/q8n_srv.py'); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); os.makedirs(S.QSLOTS, exist_ok=True); log = open(out + '/units.jsonl', 'a')
    assert not S.gpu_apps(), 'GPU busy'
    name = 'nq8g-rst-%d' % (int(time.time()) % 100000); meta = S.launch('Q', name); res = {'meta': meta, 'utc': S.now()}
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                if S.call('/health', method='GET', timeout=3)[0] == 200: break
            except Exception: pass
            time.sleep(2)
        else: raise RuntimeError('not ready')
        lg = S.sh(['docker', 'logs', name]); assert 'occamy FA kvq8 on' in lg.stdout + lg.stderr, 'native q8 path not enabled'
        ids = S.ids_load(); S.call('/completion', {'prompt': ids[:8], 'n_predict': 8, 'temperature': 0, 'id_slot': 0})
        fn = [S.QSLOTS + '/' + f for f in S.SQ.values()]
        if not all(os.path.exists(f) for f in fn):
            S.build_q8_slots(ids, log, 'Q')
            b = json.loads(open(out + '/units.jsonl').read().strip().split('\n')[-1]); nr_tokens = b['s0_prefill']['s0']['tokens'][:16]; res['no_restore_src'] = 'build prefill decode (fresh prefill, n=24, first 16)'
            res['build_s0_prefill'] = {k: v for k, v in b['s0_prefill']['s0'].items() if k != 'tokens'}
            # first restore test needs the cache the build left in slot 0 to be replaced: restore overwrites it
        else:
            nr_tokens = None; res['no_restore_src'] = 'slot files pre-existing; no fresh continuation available'
        res['n_restored'] = S.restore(0, S.SQ['s0'])
        o = S.par([(ids[0:98000], 16, 0)]); r = o['s0']; res['restore_decode'] = {k: v for k, v in r.items() if k != 'tokens'}; res['restore_tokens'] = r['tokens']
        res['no_restore_tokens'] = nr_tokens
        res['reprefill'] = (r['prompt_n'] or 0) >= 64
        res['tokens_equal'] = (nr_tokens == r['tokens']) if nr_tokens else None
        if nr_tokens and r['tokens']: res['first_diff_idx'] = next((i for i, (a, b) in enumerate(zip(nr_tokens, r['tokens'])) if a != b), None)
    finally:
        lg = S.sh(['docker', 'logs', name]); open(out + '/server-%s.log' % name, 'w').write((lg.stdout + lg.stderr)[-100000:]); S.sh(['docker', 'rm', '-f', name])
        json.dump(res, open(out + '/restore.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if 'tokens' not in k}))
if __name__ == '__main__': main()
