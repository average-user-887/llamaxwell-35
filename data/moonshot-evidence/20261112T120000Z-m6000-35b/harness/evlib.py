# Evidence harness library (H+E3). Copied/adapted from rounds/B3/{b3_deep.py,smoke.py}; originals untouched.
import os,sys,re,json,time,hashlib,subprocess,threading,http.client,io,traceback,datetime,statistics
M='<LAB>'; D=M+'/occamy35-day-20261112'; E=D+'/evidence'
GPU='<gpu>'
IMG='sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'
MODEL='/models/occamy-1.0-Q4_0-headq40b.gguf'; MODEL_HOST='<MODELS>/occamy-1.0-Q4_0-headq40b.gguf'
PORT=18493; HOST='127.0.0.1'; SCHEMA='m6000-evidence-v1'; CTX_SLOT=98304
LIBS={'A':M+'/prod-libs/libggml-cuda-gdn4-20261112.so','B':M+'/prod-libs/libggml-cuda-bundle3-20261112.so'}
SRC={'A':'85dbeed (launcher comment: llamaxwell-35 main, GDN4)','B':'2436f75 (launcher comment: day-bundle3)'}
def _env(script): return re.findall(r'-e (GGML_[A-Z0-9_]*=\S*)',open(script).read())
ENV={'A':_env(M+'/run-35b-gdn4-20261112.sh'),'B':_env(M+'/run-35b-bundle3-20261112.sh')}
def san(s):
    if not isinstance(s,str): return s
    s=s.replace(GPU,'<GPU>').replace(M,'<LAB>/m6000-llama-cpp').replace('<MODELS>','<MODELS>')
    s=s.replace('<media>/Storage','<LAB>').replace('<lab-endpoint>','<lab-endpoint>').replace('<home>','<HOME>').replace('<user>','<user>')
    s=re.sub(r'\b192\.168\.\d+\.\d+\b','<lab-ip>',s); s=re.sub(r'GPU-[0-9a-f]{8}-[0-9a-f-]{27}','<GPU>',s)
    return s
def now_utc(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')[:-4]+'Z'
def iso(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha_file(p,cache={}):
    if p in cache: return cache[p]
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    cache[p]=h.hexdigest(); return cache[p]
def append_jsonl(path,row):
    with open(path,'a') as f: f.write(san(json.dumps(row,separators=(',',':')))+'\n')
def write_atomic(path,text):
    t=path+'.tmp'; open(t,'w').write(san(text) if isinstance(text,str) else text); os.replace(t,path)
def http_json(path,body=None,method='POST',timeout=3600):
    c=http.client.HTTPConnection(HOST,PORT,timeout=timeout); 
    c.request(method,path,json.dumps(body) if body is not None else None,{'Content-Type':'application/json'}); r=c.getresponse(); d=r.read(); c.close()
    try: j=json.loads(d)
    except Exception: j={'raw':d[:300].decode(errors='replace')}
    return r.status,j
def sh(cmd,**k): return subprocess.run(cmd,capture_output=True,text=True,**k)
def gpu_busy():
    r=sh(['nvidia-smi','-i',GPU,'--query-compute-apps=pid,name','--format=csv,noheader']); return r.stdout.strip()
def buildup_running(): return sh(['docker','inspect','-f','{{.State.Running}}','build-up']).stdout.strip()=='true'

def tokenize_text(txt):
    ids=[]
    for i in range(0,len(txt),400000):
        s,j=http_json('/tokenize',{'content':txt[i:i+400000]}); assert s==200,j; ids+=j['tokens']
    return ids
def fixtures(names):
    """tokenize (via server /tokenize) and cache fixtures; returns dict name->token list. Writes fixtures_manifest.json."""
    out={}; man=json.load(open(E+'/fixtures/fixtures_manifest.json')) if os.path.exists(E+'/fixtures/fixtures_manifest.json') else {}
    for n in names:
        p=E+'/fixtures/tokens/%s.json'%n
        if os.path.exists(p): out[n]=json.load(open(p)); continue
        txt=open(E+'/fixtures/text/%s.txt'%n).read(); ids=tokenize_text(txt); json.dump(ids,open(p,'w'))
        out[n]=ids
    return out,man
def prompt_for(fx,name):
    """exact request prompts. deep: 98047 fixture tokens + [198] = 98048 input tokens; uNN: 16384; sentinel: all"""
    t=fx[name]
    if name in ('d0','d1'): return t[:98047]+[198]
    if name.startswith('u'): return t[:16384]
    return t[:256]
def write_fixture_manifest(fx):
    man={}
    for n,t in fx.items():
        pr=prompt_for(fx,n); tx=open(E+'/fixtures/text/%s.txt'%n,'rb').read()
        man[n]={'tokenized_total':len(t),'prompt_tokens':len(pr),'prompt_sha256':hashlib.sha256(json.dumps(pr).encode()).hexdigest(),'text_sha256':hashlib.sha256(tx).hexdigest(),
          'tokenization':'server /tokenize, 400000-char chunks concatenated, add_special default','template':'none (raw token-array /completion; no chat template)'}
    old=json.load(open(E+'/fixtures/fixtures_manifest.json')) if os.path.exists(E+'/fixtures/fixtures_manifest.json') else {}
    old.update(man); json.dump(old,open(E+'/fixtures/fixtures_manifest.json','w'),indent=1)

class Server:
    def __init__(s,arm,ub=1024,tag=''):
        s.arm=arm; s.ub=ub; s.sid='%s-%s-ub%d%s'%(now_utc(),'S'+arm,ub,tag); s.name='ev-'+arm.lower()+'-'+re.sub(r'[^0-9A-Za-z]','',s.sid)[-14:]
        s.dir=E+'/runs/'+s.sid; os.makedirs(s.dir,exist_ok=False)
    def start(s):
        if gpu_busy(): raise RuntimeError('GPU compute app present: '+gpu_busy())
        envs=ENV[s.arm]; e=sum([['-e',x] for x in envs],[])
        cmd=['docker','run','-d','--name',s.name,'-p','%s:%d:8080'%(HOST,PORT),'--device','nvidia.com/gpu='+GPU,
          '-v','<MODELS>:/models:ro','-v',E+'/slots:/slots','-v',LIBS[s.arm]+':/app/libggml-cuda.so:ro']+e+[IMG,
          '-m',MODEL,'--alias','qwen36-moe','--override-kv','tokenizer.ggml.pre=str:qwen35','-ngl','999','-np','2','-c','196608','-b','2048','-ub',str(s.ub),
          '--cache-ram','12288','--ctx-checkpoints','16','--checkpoint-min-step','2048','-fa','on','--host','0.0.0.0','--port','8080','--jinja','--metrics','--slot-save-path','/slots','-lv','4']
        s.cmd=cmd; s.t_start=iso()
        r=sh(cmd); write_atomic(s.dir+'/cmd.json',json.dumps({'argv':cmd,'env_vars':envs},indent=1)); write_atomic(s.dir+'/docker_run.out',r.stdout+'\n'+r.stderr)
        if r.returncode: raise RuntimeError('docker run failed: '+r.stderr[:300])
        t0=time.time()
        while time.time()-t0<900:
            try:
                st,j=http_json('/health',method='GET',timeout=3)
                if st==200 and j.get('status')=='ok': break
            except Exception: pass
            time.sleep(2)
        else: raise RuntimeError('server not ready')
        s.load_s=time.time()-t0; log=s.logs()
        s.engage={'addr_mode_1':'addr mode 1' in log,'pipe_mode_lines':[l[-120:] for l in log.splitlines() if 'pipe mode' in l][:1],
                  'n2split_lines':[l[-120:] for l in log.splitlines() if 'N2SPLIT' in l or 'n2split' in l][:2],'n2gdn_lines':[l[-120:] for l in log.splitlines() if 'N2GDN' in l or 'n2gdn' in l][:2]}
        s.kv_lines=[l[-160:] for l in log.splitlines() if 'KV buffer' in l or 'compute buffer' in l or 'RS buffer' in l][:12]
        return s
    def logs(s,since=None):
        c=['docker','logs','--timestamps']+(['--since',since] if since else [])+[s.name]; r=sh(c); return r.stdout+r.stderr
    def stop(s):
        try: write_atomic(s.dir+'/server_full.log',s.logs())
        except Exception: pass
        sh(['docker','rm','-f',s.name])

def pctl(v,p):
    v=sorted(v); 
    if not v: return None
    k=(len(v)-1)*p/100; f=int(k); c=min(f+1,len(v)-1); return v[f]+(v[c]-v[f])*(k-f)

class Stream(threading.Thread):
    """timestamped streaming /completion. ts = time.monotonic_ns() per SSE data event (one token per event)."""
    def __init__(s,name,prompt,slot,n_pred,ready_after=None,start_gate=None):
        super().__init__(daemon=True); s.name_=name; s.prompt=prompt; s.slot=slot; s.n=n_pred; s.ts=[]; s.final=None; s.err=None; s.t_send=None; s.start_gate=start_gate; s.content=''
    def run(s):
        try:
            if s.start_gate: s.start_gate()
            c=http.client.HTTPConnection(HOST,PORT,timeout=3600)
            body={'prompt':s.prompt,'n_predict':s.n,'id_slot':s.slot,'cache_prompt':True,'temperature':0,'ignore_eos':True,'stream':True}
            s.t_send=time.monotonic_ns(); c.request('POST','/completion',json.dumps(body),{'Content-Type':'application/json'}); r=c.getresponse(); s.status=r.status
            if r.status!=200: s.err='HTTP %d %s'%(r.status,r.read()[:300]); return
            for line in r:
                if line.startswith(b'data:'):
                    t=time.monotonic_ns(); j=json.loads(line[5:]); s.ts.append(t); s.content+=j.get('content','')
                    if j.get('stop'): s.final=j
            c.close()
        except Exception as ex: s.err=repr(ex)

def stream_metrics(st,t0):
    ts=[(t-t0) for t in st.ts]; f=st.final or {}; tm=f.get('timings',{})
    return {'t_send_ns':st.t_send-t0 if st.t_send else None,'ts_ns':ts,'timings':tm,'tokens_evaluated':f.get('tokens_evaluated'),'tokens_cached':f.get('tokens_cached'),
            'truncated':f.get('truncated'),'stop_type':f.get('stop_type'),'stopped_eos':f.get('stopped_eos'),'stopped_limit':f.get('stopped_limit'),'err':st.err,'n_events':len(ts)}
