# D1ATT deep ABBA on D0 saved slots (s0 98K, s1 90K). Arms in order given; per arm: own server att-srv, 1 discard + R measured reps.
# rep = restore both slots, concurrent completion (cached tokens + 1 new token, n_predict N) -> both-slot decode at ~2x94K.
import os,urllib.error,json,sys,time,threading,urllib.request,subprocess,struct,statistics
M='/media/avg-usr/Storage/m6000-llama-cpp'; R=M+'/occamy35-day-20261112/rounds/D1ATT'; D0=M+'/occamy35-day-20261112/rounds/D0'; N=M+'/occamy35-night-20261111'
GPU='GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e'; U='http://127.0.0.1:18491'
RS=M+'/run-35b-gdn4-20261112.sh'; import re
PENV=re.findall(r'-e (GGML_[A-Z0-9_]*=\S*)',open(RS).read())
ARMS={'A':(M+'/prod-libs/libggml-cuda-gdn4-20261112.so',PENV),'B':(R+'/libggml-cuda.att1.so',PENV+['GGML_CUDA_OCCAMY_FA_ADDR=1'])}
FILES=['s0_98k_off0.bin','s1_90k_off200000.bin']
def ftoks(fn):
    b=open(D0+'/slots/'+fn,'rb').read(1<<24); magic,ver,n=struct.unpack('<III',b[:12]); w=struct.unpack('<%di'%n,b[12:12+4*n]); assert w[0]==-1 and w[1]==1; return list(w[3:3+w[2]])
def post(p,d,t=3600):
    r=urllib.request.Request(U+p,json.dumps(d).encode(),{'Content-Type':'application/json'})
    try: return json.load(urllib.request.urlopen(r,timeout=t))
    except urllib.error.HTTPError as e:
        print('HTTP',e.code,p,e.read()[:500],flush=True); open(R+'/att-srv.fail.log','w').write(subprocess.run(['docker','logs','att-srv'],capture_output=True,text=True).stdout[-20000:]); subprocess.run(['docker','rm','-f','att-srv'],capture_output=True); os._exit(9)
def start(arm):
    lib,env=ARMS[arm]; e=sum([['-e',x] for x in env],[])
    subprocess.run(['docker','rm','-f','att-srv'],capture_output=True)
    subprocess.run(['docker','run','-d','--name','att-srv','-p','127.0.0.1:18491:8080','--device','nvidia.com/gpu='+GPU,'-v','/media/avg-usr/Storage/uroboros-models:/models:ro',
      '-v',N+'/app:/app:ro','-v',lib+':/app/libggml-cuda.so:ro','-v',D0+'/slots:/slots:ro','-e','LD_LIBRARY_PATH=/app']+e+['--entrypoint','/app/llama-server','nvidia/cuda:12.8.1-devel-ubuntu24.04',
      '-m','/models/occamy-1.0-Q4_0-headq40b.gguf','--alias','qwen36-moe','--override-kv','tokenizer.ggml.pre=str:qwen35','-ngl','999','-np','2','-c','196608','-b','2048','-ub','1024','-fa','on',
      '--host','0.0.0.0','--port','8080','--no-warmup','--slot-save-path','/slots','--cache-ram','0'],check=True,capture_output=True)
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(U+'/health',timeout=3)).get('status')=='ok': return
        except Exception: pass
        time.sleep(2)
    sys.exit('server not ready')
def rep(T,n):
    for i,fn in enumerate(FILES): post('/slots/%d?action=restore'%i,{'filename':fn})
    o={}
    def c(i): o[i]=post('/completion',{'prompt':T[i]+[198],'n_predict':n,'id_slot':i,'cache_prompt':True,'temperature':0,'ignore_eos':True})['timings']
    th=[threading.Thread(target=c,args=(i,)) for i in range(2)]; t0=time.time(); [x.start() for x in th]; [x.join() for x in th]
    return {'wall':time.time()-t0,**{'s%d'%i:{k:o[i][k] for k in ('prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second')} for i in o}}
if __name__=='__main__':
    order=sys.argv[1]; reps=int(sys.argv[2]); n=int(sys.argv[3]); T=None
    out=[]
    for k,arm in enumerate(order):
        start(arm)
        if T is None:
            t=open(D0+'/deep_text.txt').read(); ids=[]
            for i in range(0,len(t),400000): ids+=post('/tokenize',{'content':t[i:i+400000]})['tokens']
            T=[ids[0:98000],ids[200000:290000]]; print('prompt toks',[len(x) for x in T],flush=True)
        lg=subprocess.run(['docker','logs','att-srv'],capture_output=True,text=True); log=lg.stdout+lg.stderr
        print(arm,'addr' if 'addr mode 1' in log else 'noaddr',[l for l in log.splitlines() if 'pipe mode' in l][:1],flush=True)
        for r in range(reps+1):
            x=rep(T,n); x.update(arm=arm,pos=k,rep=r,warm=(r==0)); out.append(x); print(json.dumps(x),flush=True)
        subprocess.run(['docker','rm','-f','att-srv'],capture_output=True)
    json.dump(out,open(R+'/deep_abba.json','w'),indent=1)
    for arm in 'AB':
        v=[x for x in out if x['arm']==arm and not x['warm']]
        p=[statistics.mean([x['s0']['predicted_ms']/x['s0']['predicted_n'],x['s1']['predicted_ms']/x['s1']['predicted_n']]) for x in v]
        print(arm,'n',len(v),'per-slot period ms mean %.3f sd %.3f'%(statistics.mean(p),statistics.pstdev(p)),'agg t/s %.2f'%statistics.mean([x['s0']['predicted_per_second']+x['s1']['predicted_per_second'] for x in v]))
