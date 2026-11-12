#!/usr/bin/env python3
# e4_run.py <B|A>: E4 diagnostic. For mode in plain, prof: own server ev4-<arm>-<mode> (launcher flags/env/lib), 4 phases, timings from /completion.
# nvprof method as rounds/D0/srv.sh prof (--csv --print-gpu-trace, whole run, phases separated by idle gaps). GPU lock held externally.
import os,sys,re,json,time,struct,subprocess,threading,urllib.request,urllib.error,datetime
arm=sys.argv[1]; M='<LAB>'; D=M+'/occamy35-day-20261112'; N=M+'/occamy35-night-20261111'
GPU='<gpu>'; IMG='nvidia/cuda:12.8.1-devel-ubuntu24.04'
RS={'B':M+'/run-35b-bundle3-20261112.sh','A':M+'/run-35b-gdn4-20261112.sh'}[arm]
LIB={'B':M+'/prod-libs/libggml-cuda-bundle3-20261112.so','A':M+'/prod-libs/libggml-cuda-gdn4-20261112.so'}[arm]
ts=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'); OUT=f'{D}/evidence/E4/{ts}-{arm}'
os.makedirs(OUT)  # refuses to overwrite
ENV=re.findall(r'-e (GGML_[A-Z0-9_]*=\S*)',open(RS).read())
PORT=18495
bu=subprocess.run(['docker','ps','--filter','name=^build-up$','--format','{{.Names}} {{.Status}}'],capture_output=True,text=True).stdout.strip()
rec={'arm':arm,'lib':LIB,'launcher':RS,'env':ENV,'build_up_running_at_start':bool(bu),'build_up':bu,'loadavg':open('/proc/loadavg').read().strip(),'phases':{}}
def sh(c,**k): return subprocess.run(c,capture_output=True,text=True,**k)
def post(p,d,t=1800):
    r=urllib.request.Request(f'http://127.0.0.1:{PORT}'+p,json.dumps(d).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r,timeout=t))
def ftoks(fn):
    b=open(D+'/rounds/D0/slots/'+fn,'rb').read(1<<24); m,v,n=struct.unpack('<III',b[:12]); w=struct.unpack('<%di'%n,b[12:12+4*n]); assert w[0]==-1 and w[1]==1; return list(w[3:3+w[2]])
FILES=['s0_98k_off0.bin','s1_90k_off200000.bin']; T=[ftoks(f) for f in FILES]
def tim(r): return {k:r['timings'][k] for k in ('prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second')}
def comp(slot,prompt,n,out,key,cache=False):
    out[key]=tim(post('/completion',{'prompt':prompt,'n_predict':n,'id_slot':slot,'cache_prompt':cache,'temperature':0,'ignore_eos':True}))
def par(fs):
    th=[threading.Thread(target=f) for f in fs]; [t.start() for t in th]; [t.join() for t in th]
def run_mode(mode):
    cn=f'ev4-{arm}-{mode}'; od=f'{OUT}/{mode}'; os.makedirs(od)
    e=sum([['-e',x] for x in ENV],[])
    srv=['-m','/models/occamy-1.0-Q4_0-headq40b.gguf','--alias','qwen36-moe','--override-kv','tokenizer.ggml.pre=str:qwen35','-ngl','999','-np','2','-c','196608','-b','2048','-ub','1024','--cache-ram','12288','--ctx-checkpoints','16','--checkpoint-min-step','2048','-fa','on','--host','0.0.0.0','--port','8080','--jinja','--metrics','-lv','4']
    if mode=='prof': ep=['--entrypoint','/usr/local/cuda/bin/nvprof']; pre=['--csv','--print-gpu-trace','--log-file',f'/out/prof.csv','/app/llama-server']
    else: ep=['--entrypoint','/app/llama-server']; pre=[]
    cmd=['docker','run','-d','--name',cn,'-p',f'127.0.0.1:{PORT}:8080','--device','nvidia.com/gpu='+GPU,'-v','<MODELS>:/models:ro','-v',N+'/app:/app:ro','-v',LIB+':/app/libggml-cuda.so:ro','-v',D+'/rounds/D0/slots:/slots:ro','-v',od+':/out','-e','LD_LIBRARY_PATH=/app']+e+ep+[IMG]+pre+srv+['--slot-save-path','/slots']
    open(f'{od}/cmd.txt','w').write(' '.join(cmd)+'\n')
    r=sh(cmd); open(f'{od}/docker_run.rc','w').write(str(r.returncode)); open(f'{od}/docker_run.out','w').write(r.stdout+r.stderr)
    if r.returncode: return {'error':'docker run failed'}
    ok=False
    for _ in range(400):
        try:
            if json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3)).get('status')=='ok': ok=True;break
        except Exception: pass
        time.sleep(3)
    ph={}
    if ok:
        try:
            t=time.time(); o={}
            comp(0,T[0][5000:9096],1,o,'p',False); ph['1_prefill4k']=o['p']; ph['1_prefill4k']['wall_s']=time.time()-t; time.sleep(5)
            o={}; comp(0,list(range(1000,1020)),64,o,'a'); ph['2_n1_shallow']=o['a']; time.sleep(5)
            o={}; par([lambda:comp(0,list(range(1000,1020)),64,o,'s0'),lambda:comp(1,list(range(2000,2020)),64,o,'s1')]); ph['3_n2_shallow']=o; time.sleep(5)
            for i in range(2): post(f'/slots/{i}?action=restore',{'filename':FILES[i]})
            o={}; par([lambda:comp(0,T[0]+[198],64,o,'s0',True),lambda:comp(1,T[1]+[198],64,o,'s1',True)]); ph['4_n2_deep']=o; time.sleep(5)
        except Exception as ex: ph['error']=repr(ex)
    else: ph['error']='server not ready'
    open(f'{od}/server.log','w').write(sh(['docker','logs',cn]).stdout+sh(['docker','logs',cn]).stderr)
    r=sh(['docker','stop','-t','180',cn]); open(f'{od}/stop.rc','w').write(str(r.returncode))
    open(f'{od}/server_after_stop.log','w').write(sh(['docker','logs','--tail','60',cn]).stdout)
    sh(['docker','rm','-f',cn]); json.dump(ph,open(f'{od}/timings.json','w'),indent=1); return ph
for mode in ('plain','prof'):
    rec['phases'][mode]=run_mode(mode); json.dump(rec,open(f'{OUT}/record.json','w'),indent=1)
rec['build_up_running_at_end']=bool(sh(['docker','ps','--filter','name=^build-up$','-q']).stdout.strip())
json.dump(rec,open(f'{OUT}/record.json','w'),indent=1); print('done',OUT)
