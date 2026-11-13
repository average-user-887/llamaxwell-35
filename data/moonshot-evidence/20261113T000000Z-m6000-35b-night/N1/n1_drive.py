import sys, json, time, http.client, datetime
M='<LAB>'
out, arm = sys.argv[1], sys.argv[2]
def call(path, body, timeout=3600):
    c=http.client.HTTPConnection('127.0.0.1',18494,timeout=timeout); c.request('POST',path,json.dumps(body),{'Content-Type':'application/json'}); r=c.getresponse(); d=r.read(); c.close(); return r.status,json.loads(d)
tok=json.load(open(M+'/codex-investigation-20261112/fixture_tokens.json'))
base=tok[200000:290000]; fresh=tok[290000:]
log=[]; GAP=5
def step(cell, phase, f):
    t0=time.time(); r=f(); t1=time.time()
    log.append(dict(cell=cell, phase=phase, t0=t0, t1=t1, wall=t1-t0, **r)); print(cell, phase, round(t1-t0,2), r, flush=True); time.sleep(GAP)
def restore():
    st,j=call('/slots/0?action=restore',{'filename':'s1_90k_off200000.bin'}); assert st==200 and j.get('n_restored')==90000,(st,j); return {'n_restored':j['n_restored'],'srv_ms':j.get('t_ms')}
def comp(n):
    def f():
        st,j=call('/completion',{'prompt':base+fresh[:n],'id_slot':0,'n_predict':1,'cache_prompt':True,'temperature':0,'ignore_eos':True,'return_tokens':True}); assert st==200,j
        t=j['timings']; return {'prompt_n':t['prompt_n'],'prompt_ms':t['prompt_ms'],'predicted_n':t['predicted_n'],'predicted_ms':t['predicted_ms'],'n_ctx_cached':j.get('tokens_cached')}
    return f
def probe(n):
    def f():
        st,j=call('/completion',{'prompt':base+fresh[:n]+[fresh[n+7]],'id_slot':0,'n_predict':1,'cache_prompt':True,'temperature':0,'ignore_eos':True}); assert st==200,j
        t=j['timings']; return {'prompt_n':t['prompt_n'],'prompt_ms':t['prompt_ms'],'predicted_n':t['predicted_n']}
    return f
cells=[('warm',1024),('r1_1K',1024),('r1_4K',4096),('r2_1K',1024),('r2_4K',4096)]
for c,n in cells:
    step(c,'restore',restore); step(c,'prefill',comp(n)); step(c,'probe',probe(n))
json.dump(log,open(out+'/steps.json','w'),indent=1)
