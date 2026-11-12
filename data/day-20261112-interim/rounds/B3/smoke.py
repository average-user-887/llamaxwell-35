# smoke.py <A|B> <cases a..e>   throwaway b3-smoke on 127.0.0.1:18492, prod server args; lock held externally
import os,sys,json,time,threading,urllib.request,urllib.error,subprocess,re,struct
M='/media/avg-usr/Storage/m6000-llama-cpp'; W=M+'/occamy35-day-20261112/rounds/B3'; D0=M+'/occamy35-day-20261112/rounds/D0'; N=M+'/occamy35-night-20261111'
GPU='GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e'; U='http://127.0.0.1:18492'; C='b3-smoke'
arm=sys.argv[1]; cases=sys.argv[2]
PENV=re.findall(r'-e (GGML_[A-Z0-9_]*=\S*)',open(M+'/run-35b-gdn4-20261112.sh').read())
if arm=='A': lib,env=M+'/prod-libs/libggml-cuda-gdn4-20261112.so',PENV
else: lib,env=W+'/libggml-cuda.b3.so',PENV+['GGML_CUDA_OCCAMY_N2SPLIT=1','GGML_CUDA_OCCAMY_FA_ADDR=1','GGML_CUDA_OCCAMY_N2GDN=1']
FILES=['s0_98k_off0.bin','s1_90k_off200000.bin']; res={}
def post(p,d,t=1800):
    r=urllib.request.Request(U+p,json.dumps(d).encode(),{'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(r,timeout=t) as f: return f.status,json.load(f)
    except urllib.error.HTTPError as e: return e.code,{'err':e.read()[:300].decode()}
def sane(t): return len(t.strip())>=20 and len(set(t))>8
subprocess.run(['docker','rm','-f',C],capture_output=True)
e=sum([['-e',x] for x in env],[])
subprocess.run(['docker','run','-d','--name',C,'-p','127.0.0.1:18492:8080','--device','nvidia.com/gpu='+GPU,'-v','/media/avg-usr/Storage/uroboros-models:/models:ro','-v',N+'/app:/app:ro','-v',lib+':/app/libggml-cuda.so:ro','-v',D0+'/slots:/slots:ro','-e','LD_LIBRARY_PATH=/app']+e+['--entrypoint','/app/llama-server','nvidia/cuda:12.8.1-devel-ubuntu24.04',
 '-m','/models/occamy-1.0-Q4_0-headq40b.gguf','--alias','qwen36-moe','--override-kv','tokenizer.ggml.pre=str:qwen35','-ngl','999','-np','2','-c','196608','-b','2048','-ub','1024','--cache-ram','12288','--ctx-checkpoints','16','--checkpoint-min-step','2048','-fa','on','--host','0.0.0.0','--port','8080','--jinja','--metrics','--slot-save-path','/slots','-lv','4'],check=True,capture_output=True)
for _ in range(300):
    try:
        if json.load(urllib.request.urlopen(U+'/health',timeout=3)).get('status')=='ok': break
    except Exception: pass
    time.sleep(2)
else: sys.exit('server not ready')
t=open(D0+'/deep_text.txt').read(); ids=[]
for i in range(0,len(t),400000): ids+=post('/tokenize',{'content':t[i:i+400000]})[1]['tokens']
print('ntoks',len(ids),flush=True)
T=[ids[0:98000],ids[200000:290000]]; fresh=ids[290000:306384] if len(ids)>=306384 else ids[110000:126384]
def restore(i): s,_=post('/slots/%d?action=restore'%i,{'filename':FILES[i]}); assert s==200,('restore',i,s)
def comp(i,toks,n,out,k):
    s,j=post('/completion',{'prompt':toks,'n_predict':n,'id_slot':i,'cache_prompt':True,'temperature':0,'ignore_eos':True}); out[k]=(s,j)
def rec(name,ok,info): res[name]=(ok,info); print(name,'PASS' if ok else 'FAIL',info,flush=True)
def stream_cancel(i,toks,n=512,chunks=8):
    r=urllib.request.Request(U+'/completion',json.dumps({'prompt':toks,'n_predict':n,'id_slot':i,'cache_prompt':True,'temperature':0,'ignore_eos':True,'stream':True}).encode(),{'Content-Type':'application/json'})
    f=urllib.request.urlopen(r,timeout=900); st=f.status; got=0
    for line in f:
        if line.startswith(b'data:'):
            got+=1
            if got>=chunks: break
    f.close(); return st,got
try:
  if 'a' in cases:
    for i in range(2): restore(i)
    o={}; th=[threading.Thread(target=comp,args=(i,T[i]+[198],128,o,i)) for i in range(2)]; [x.start() for x in th]; [x.join() for x in th]
    rec('a',all(o[i][0]==200 and o[i][1]['timings']['predicted_n']==128 and sane(o[i][1]['content']) for i in o),[ (o[i][0],round(o[i][1]['timings']['predicted_per_second'],2),o[i][1]['content'][:60]) for i in o])
  if 'b' in cases:
    for i in range(2): restore(i)
    o={}; th=[threading.Thread(target=comp,args=(1,T[1]+[198],128,o,1)),threading.Thread(target=lambda:(time.sleep(2),comp(0,fresh,64,o,0)))]; [x.start() for x in th]; [x.join() for x in th]
    rec('b',all(o[i][0]==200 and sane(o[i][1]['content']) for i in o),[(o[i][0],o[i][1]['timings']['prompt_n'],round(o[i][1]['timings']['predicted_per_second'],2),o[i][1]['content'][:60]) for i in o])
  if 'c' in cases:
    s,g=stream_cancel(0,fresh[:2000]); o={}; comp(0,fresh[:2000]+[198],48,o,0)
    rec('c',s==200 and o[0][0]==200 and sane(o[0][1]['content']),[s,g,o[0][0],o[0][1]['content'][:60]])
  if 'd' in cases:
    s,j=post('/v1/chat/completions',{'model':'qwen36-moe','messages':[{'role':'user','content':'Name three primary colors and one fact about each.'}],'max_tokens':160,'temperature':0})
    txt=j['choices'][0]['message'].get('content') or j['choices'][0]['message'].get('reasoning_content') or '' if s==200 else ''
    rec('d',s==200 and sane(txt),[s,txt[:80]])
  if 'e' in cases:
    restore(1); s,g=stream_cancel(1,T[1]+[198],512,6); restore(1); o={}; comp(1,T[1]+[198],64,o,1)
    rec('e',s==200 and o[1][0]==200 and sane(o[1][1]['content']),[s,g,o[1][0],o[1][1]['content'][:60]])
finally:
  lg=subprocess.run(['docker','logs',C],capture_output=True,text=True); open(W+'/smoke_%s.log'%arm,'w').write(lg.stdout+lg.stderr)
  subprocess.run(['docker','rm','-f',C],capture_output=True)
json.dump({k:[v[0],str(v[1])] for k,v in res.items()},open(W+'/smoke_%s.json'%arm,'w'))
