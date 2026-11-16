import json,sys,time,threading,urllib.request as U,glob,os
port,mode,np_,maxd,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
B='http://127.0.0.1:%s'%port
def post(p,b,t=3600):
    r=U.Request(B+p,json.dumps(b).encode(),{'Content-Type':'application/json'})
    with U.urlopen(r,timeout=t) as f: return json.loads(f.read())
def corp(rev):
    fs=sorted(glob.glob('/usr/lib/python3.12/*.py'),reverse=rev)
    txt=''.join(open(f,errors='ignore').read() for f in fs)
    return post('/tokenize',{'content':txt[:maxd*6+60000]})['tokens']
TA=corp(False); TB=corp(True)
TG="Write a Python module implementing an LRU cache with TTL, thread safety, type hints and docstrings, then explain its complexity in detail."
R={'mode':mode,'rows':[]}
def sv(): json.dump(R,open(out,'w'),indent=1)
def one(slot,prompt,n,label):
    t=time.time()
    r=post('/completion',{'prompt':prompt,'n_predict':n,'id_slot':slot,'cache_prompt':False,'temperature':0,'ignore_eos':True})
    m=r['timings']; row={'row':label,'slot':slot,'prompt_n':m['prompt_n'],'pp_tps':round(m['prompt_per_second'],1),'tg_tps':round(m['predicted_per_second'],2),'n':m['predicted_n'],'draft_n':m.get('draft_n'),'draft_acc':m.get('draft_n_accepted'),'wall_s':round(time.time()-t,1)}
    R['rows'].append(row); sv(); print(row,flush=True); return row
def tryone(*a):
    try: return one(*a)
    except Exception as e:
        R['rows'].append({'row':a[3],'error':str(e)[:200]}); sv(); print('ERR',a[3],e,flush=True)
depths=[d for d in (52000,80000) if d<=maxd]
# 1-slot phase
tryone(0,TG,192,'1s_d26')
tryone(0,TA[:4096],1,'1s_pp4k')
for d in depths: tryone(0,TA[:d],128,'1s_d%d'%d)
# 2-slot concurrent clean-window decode
def stream(slot,prompt,n,rec,stop):
    body={'prompt':prompt,'n_predict':n,'id_slot':slot,'cache_prompt':False,'temperature':0,'ignore_eos':True,'stream':True,'return_tokens':True}
    r=U.Request(B+'/completion',json.dumps(body).encode(),{'Content-Type':'application/json'})
    ev=[]; last={}
    try:
        f=U.urlopen(r,timeout=3600)
        for line in f:
            if stop.is_set(): break
            if not line.startswith(b'data:'): continue
            j=json.loads(line[5:]); k=len(j.get('tokens') or [])
            if k: ev.append((time.time(),k))
            if j.get('stop'): last=j.get('timings',{}); break
        f.close()
    except Exception as e: rec['err']=str(e)[:200]
    rec['ev']=ev; rec['timings']=last
def two(label,pa,pb,n0,n1):
    r0={};r1={};stop=threading.Event()
    t0=threading.Thread(target=stream,args=(0,pa,n0,r0,stop)); t1=threading.Thread(target=stream,args=(1,pb,n1,r1,stop))
    t0.start(); t1.start(); t1.join(); stop.set(); t0.join()
    row={'row':label}
    try:
        e1=r1['ev']; w0,w1=e1[0][0],e1[-1][0]; tk0=sum(k for t,k in r0['ev'] if w0<=t<=w1)
        tm=r1['timings']
        row.update(s1_tg_tps=round(tm['predicted_per_second'],2),s1_n=tm['predicted_n'],s1_draft_n=tm.get('draft_n'),s1_draft_acc=tm.get('draft_n_accepted'),s1_pp_tps=round(tm['prompt_per_second'],1),
                   s0_window_tps=round(tk0/(w1-w0),2),window_s=round(w1-w0,1),s0_prompt_n=(r0.get('timings') or {}).get('prompt_n'))
        row['agg_tps']=round(row['s1_tg_tps']+row['s0_window_tps'],2)
    except Exception as e: row['error']=str(e)[:200]+' '+str(r0.get('err'))+str(r1.get('err'))
    R['rows'].append(row); sv(); print(row,flush=True)
    time.sleep(3)
if np_>=2:
    two('2s_d26',TG,TG+' (variant)',900,256)
    for d in depths: two('2s_d%d'%d,TA[:d],TB[:d],2500,256)
R['done']=True; sv()
