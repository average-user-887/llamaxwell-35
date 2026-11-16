# MV2 client. usage: mv2_client.py HOST:PORT tag np pn npred reps out.json [ref_dir]
# long mode (pn>=20000): per slot cold run, then `reps` cached repeats (decode-only timing), then a greedy n_probs=5 run (96 tok).
# short mode (pn<20000): fixtures corp (stdlib corpus) and rand (seeded pseudo-random hex words), greedy, n_probs=5, plus teacher-forced probe vs ref_dir/result.json tokens.
import json,sys,time,threading,urllib.request as U,glob,random,os
hp,tag,np_,pn,npred,reps,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7]
ref=sys.argv[8] if len(sys.argv)>8 else None
B='http://'+hp
def post(p,b,t=7200):
    r=U.Request(B+p,json.dumps(b).encode(),{'Content-Type':'application/json'})
    with U.urlopen(r,timeout=t) as f: return json.loads(f.read())
def corp(rev,n):
    fs=sorted(glob.glob('/usr/lib/python3.12/*.py'),reverse=rev)
    txt=''.join(open(f,errors='ignore').read() for f in fs)
    return post('/tokenize',{'content':txt[:n*6+60000]})['tokens'][:n]
def rnd(n):
    r=random.Random(7); txt=' '.join('%08x'%r.getrandbits(32) for _ in range(n))
    return post('/tokenize',{'content':txt})['tokens'][:n]
TK=('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second','draft_n','draft_n_accepted')
R={'tag':tag,'np':np_,'pn':pn,'n_predict':npred,'reps':reps,'runs':{}}
def sv(): json.dump(R,open(out,'w'),indent=1)
def stream(prompt,slot,n,extra=None):
    body={'prompt':prompt,'n_predict':n,'id_slot':slot,'cache_prompt':True,'ignore_eos':True,'stream':True,'return_tokens':True,'temperature':0}
    if extra: body.update(extra)
    t0=time.time(); ev=[]; toks=[]; last={}; probs=[]
    rec={}
    try:
        f=U.urlopen(U.Request(B+'/completion',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=7200)
        for line in f:
            if not line.startswith(b'data:'): continue
            j=json.loads(line[5:]); tk=j.get('tokens') or []
            if tk: ev.append((round(time.time()-t0,4),len(tk))); toks+=tk
            if j.get('completion_probabilities'): probs+=j['completion_probabilities']
            if j.get('stop'): last=j.get('timings',{}); break
        f.close()
    except Exception as e: rec['err']=str(e)[:200]
    rec.update(t_first_s=ev[0][0] if ev else None,latency_s=round(time.time()-t0,2),n_out=len(toks),out_tokens=toks,ev=ev,timings={k:last.get(k) for k in TK})
    if last.get('draft_n'): rec['accept_rate']=round(last['draft_n_accepted']/last['draft_n'],4)
    if probs: rec['probs']=probs
    return rec
def par(fn):
    res={}; ths=[threading.Thread(target=lambda s=s: res.__setitem__(s,fn(s))) for s in range(np_)]
    [t.start() for t in ths]; [t.join() for t in ths]; return res
if pn>=20000:
    P=[corp(False,pn),corp(True,pn)][:np_]
    for name,kw in [('cold',{}),*[('rep%d'%i,{}) for i in range(reps)]]:
        t0=time.time(); r=par(lambda s: stream(P[s],s,npred)); R['runs'][name]={'wall_s':round(time.time()-t0,2),'slots':{str(s):v for s,v in r.items()}}
        if np_==2:
            a,b=r[0],r[1]
            try:
                w0=max(a['t_first_s'],b['t_first_s']); w1=min(a['ev'][-1][0],b['ev'][-1][0])
                if w1>w0: R['runs'][name]['agg_window_tps']=round(sum(k for s in (a,b) for t,k in s['ev'] if w0<t<=w1)/(w1-w0),2)
            except Exception as e: pass
        sv()
    r=par(lambda s: stream(P[s],s,96,{'n_probs':5})); R['runs']['probs96']={'slots':{str(s):v for s,v in r.items()}}
else:
    fx={'corp':corp(False,pn),'rand':rnd(pn)}
    refd=json.load(open(ref+'/result.json')) if ref else None
    for nm,p in fx.items():
        r=stream(p,0,npred,{'n_probs':5,'cache_prompt':False}); R['runs'][nm]=r; sv()
        r2=stream(p,0,npred,{'cache_prompt':False}); R['runs'][nm+'_noprobs']={k:r2.get(k) for k in ('out_tokens','timings','accept_rate','n_out')}; sv()
        if refd:
            rt=refd['runs'][nm]['out_tokens']; at=r['out_tokens']
            d=next((i for i in range(min(len(rt),len(at))) if rt[i]!=at[i]),None)
            R['runs'][nm]['first_diff_vs_ref']=d
            if d is not None:
                R['runs'][nm]['teacher_forced']=stream(p+rt[:d],0,1,{'n_probs':5,'cache_prompt':False})
    for sd in (1234,99):
        r=stream(fx['corp'],0,npred,{'cache_prompt':False,'temperature':1.5,'top_k':0,'top_p':1.0,'min_p':0.0,'seed':sd}); R['runs']['hot%d'%sd]={k:r.get(k) for k in ('n_out','timings','accept_rate','latency_s')}; sv()
    sv()
R['done']=True; sv()
