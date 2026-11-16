# MV1 client: concurrent per-slot long prompts (public stdlib corpus), greedy or server-default sampling.
# usage: mv1_client.py HOST:PORT tag np prompt_tokens n_predict out.json sampling(0|1)
import json,sys,time,threading,urllib.request as U,glob
hp,tag,np_,pn,npred,out,samp=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6],int(sys.argv[7])
B='http://'+hp
def post(p,b,t=3600):
    r=U.Request(B+p,json.dumps(b).encode(),{'Content-Type':'application/json'})
    with U.urlopen(r,timeout=t) as f: return json.loads(f.read())
def get(p):
    with U.urlopen(B+p,timeout=30) as f: return json.loads(f.read())
def corp(rev):
    fs=sorted(glob.glob('/usr/lib/python3.12/*.py'),reverse=rev)
    txt=''.join(open(f,errors='ignore').read() for f in fs)
    return post('/tokenize',{'content':txt[:pn*6+60000]})['tokens'][:pn]
P=[corp(False),corp(True)][:np_]
R={'tag':tag,'np':np_,'prompt_tokens':pn,'n_predict':npred,'sampling':'server-default+seed' if samp else 'greedy(temp0)','slots':{}}
try: R['props_default_gen']=get('/props').get('default_generation_settings')
except Exception as e: R['props_err']=str(e)[:100]
def sv(): json.dump(R,open(out,'w'),indent=1)
def run(slot,rec):
    body={'prompt':P[slot],'n_predict':npred,'id_slot':slot,'cache_prompt':False,'ignore_eos':True,'stream':True,'return_tokens':True}
    if samp: body['seed']=1234
    else: body['temperature']=0
    t0=time.time(); ev=[]; toks=[]; last={}
    try:
        r=U.Request(B+'/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
        f=U.urlopen(r,timeout=7200)
        for line in f:
            if not line.startswith(b'data:'): continue
            j=json.loads(line[5:]); tk=j.get('tokens') or []
            if tk: ev.append((round(time.time()-t0,4),len(tk))); toks+=tk
            if j.get('stop'): last=j.get('timings',{}); break
        f.close()
    except Exception as e: rec['err']=str(e)[:200]
    rec['t_first_s']=ev[0][0] if ev else None; rec['latency_s']=round(time.time()-t0,2); rec['out_tokens']=toks; rec['n_out']=len(toks)
    rec['timings']={k:last.get(k) for k in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second','draft_n','draft_n_accepted')}
    if last.get('draft_n'): rec['accept_rate']=round(last['draft_n_accepted']/last['draft_n'],4)
    rec['ev']=ev
    rec['occupancy']=round((last.get('prompt_n',0)+len(toks))/84224,4) if last else None
ths=[]
for s in range(np_):
    R['slots'][str(s)]={}; ths.append(threading.Thread(target=run,args=(s,R['slots'][str(s)])))
t0=time.time()
for t in ths: t.start()
for t in ths: t.join()
R['wall_s']=round(time.time()-t0,2)
# concurrent decode window agg: both slots past first token
try:
    if np_==2:
        a,b=R['slots']['0'],R['slots']['1']
        w0=max(a['t_first_s'],b['t_first_s']); w1=min(a['ev'][-1][0],b['ev'][-1][0])
        if w1>w0: R['agg_window_tps']=round(sum(k for s in (a,b) for t,k in s['ev'] if w0<t<=w1)/(w1-w0),2); R['agg_window_s']=round(w1-w0,1)
except Exception as e: R['agg_err']=str(e)[:100]
try: R['slots_after']=[{k:s.get(k) for k in ('id','n_ctx','is_processing')} for s in get('/slots')]
except Exception as e: R['slots_err']=str(e)[:100]
R['done']=True; sv()
