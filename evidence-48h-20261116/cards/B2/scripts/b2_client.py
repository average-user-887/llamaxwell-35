import json, sys, time, threading, http.client, glob
port, np_, depths, out, phase = int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')], sys.argv[4], sys.argv[5]
NG = 192
def setph(s):
    open(phase, 'w').write(s + '\n')
def post(p, b, t=3600):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=t); c.request('POST', p, json.dumps(b), {'Content-Type': 'application/json'}); r = c.getresponse(); d = json.loads(r.read()); c.close(); return d
def corp(rev):
    fs = sorted(glob.glob('/usr/lib/python3.12/*.py'), reverse=rev)
    txt = ''.join(open(f, errors='ignore').read() for f in fs)
    return post('/tokenize', {'content': txt[:max(depths) * 6 + 60000]})['tokens']
TA = corp(False); TB = corp(True)
R = {'rows': []}
def sv(): json.dump(R, open(out, 'w'), indent=1)
def req(slot, prompt, n, label, flip_label=None, rec=None):
    """stream /completion; if flip_label, set phase when first token arrives; else phase is managed by caller."""
    t0 = time.time(); toks = []; tm = None; tfirst = None
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=3600)
    c.request('POST', '/completion', json.dumps({'prompt': prompt, 'n_predict': n, 'id_slot': slot, 'cache_prompt': True, 'temperature': 0, 'ignore_eos': True, 'stream': True, 'return_tokens': True}), {'Content-Type': 'application/json'})
    r = c.getresponse()
    for line in r:
        line = line.decode(errors='replace').strip()
        if not line.startswith('data:'): continue
        try: j = json.loads(line[5:])
        except Exception: continue
        if j.get('tokens'):
            if tfirst is None:
                tfirst = time.time()
                if flip_label: setph(flip_label)
            toks += j['tokens']
        if j.get('timings'): tm = j['timings']
    c.close()
    row = {'row': label, 'slot': slot, 'prompt_len': len(prompt), 'wall_s': round(time.time() - t0, 1), 'n_gen': len(toks)}
    if tm: row.update({k: tm.get(k) for k in ('prompt_n', 'prompt_per_second', 'predicted_n', 'predicted_per_second', 'draft_n', 'draft_n_accepted')})
    if rec is not None: rec.update(row); rec['toks'] = toks
    R['rows'].append(row); sv(); print(row, flush=True)
    return toks
try:
    if np_ == 1:
        d0 = depths[0]; prompt = TA[:d0]
        gen = req(0, prompt, NG, 's1_d%d' % d0, flip_label='s1_d%d' % d0); setph('idle'); time.sleep(0.6)
        prompt = prompt + gen
        for d in depths[1:]:
            prompt = prompt + TA[len(prompt):d]
            gen = req(0, prompt, NG, 's1_d%d' % d, flip_label='s1_d%d' % d); setph('idle'); time.sleep(0.6)
            prompt = prompt + gen
    else:
        d0 = depths[0]
        pa = TA[:d0]; ga = req(0, pa, NG, 's1_d%d' % d0, flip_label='s1_d%d' % d0); setph('idle'); time.sleep(0.6)
        pb = TB[:d0]; gb = req(1, pb, 8, 'prefill_slot1'); time.sleep(0.6)
        pa = pa + ga + TA[d0:d0 + 64]; pb = pb + gb + TB[d0:d0 + 64]
        setph('s2_d%d' % d0); time.sleep(0.4)
        th = [threading.Thread(target=req, args=(0, pa, NG, 's2_slot0')), threading.Thread(target=req, args=(1, pb, NG, 's2_slot1'))]
        [t.start() for t in th]; [t.join() for t in th]
        setph('idle'); time.sleep(0.8)
except Exception as e:
    R['error'] = repr(e); sv(); print('ERR', e, flush=True)
setph('idle')
