#!/usr/bin/env python3
"""E9 part 1: A=default verbosity, B=-lv 4; single slot ~98K restored context, 64 tokens/unit. usage: e9_run.py <outdir> <order>"""
import sys, os, json, time, subprocess, http.client
M='<LAB>'; E=M+'/occamy35-day-20261112/evidence'; D=E+'/E9'; PORT=18494
GPU=os.environ['E9_GPU']; LV={'A':'default','B':'4'}; NU=int(os.environ.get('N_UNITS','4'))
def sh(c): return subprocess.run(c,capture_output=True,text=True)
def apps(): return sh(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader']).stdout.strip()
def call(path,body=None,method='POST',timeout=900):
    c=http.client.HTTPConnection('127.0.0.1',PORT,timeout=timeout); c.request(method,path,json.dumps(body) if body is not None else None,{'Content-Type':'application/json'})
    r=c.getresponse(); d=r.read(); c.close()
    try: return r.status,json.loads(d)
    except Exception: return r.status,{}
SLOTF='e9_np1_98k.bin'
d0=json.load(open(E+'/fixtures/tokens/d0.json')); PROMPT=d0[:98047]+[198]
def unit():
    t=time.monotonic(); st,j=call('/slots/0?action=restore',{'filename':SLOTF}); assert st==200,(st,j); t_r=time.monotonic()
    st,j=call('/completion',{'prompt':PROMPT,'n_predict':64,'id_slot':0,'cache_prompt':True,'temperature':0,'ignore_eos':True}); t2=time.monotonic()
    tm=j.get('timings',{}); return dict(restored=j.get('tokens_cached'),restore_s=t_r-t,wall_s=t2-t,gen_wall_s=t2-t_r,prompt_n=tm.get('prompt_n'),cache_n=tm.get('cache_n'),
        predicted_n=tm.get('predicted_n'),predicted_ms=tm.get('predicted_ms'),ms_per_tok=tm.get('predicted_per_token_ms'),prompt_ms=tm.get('prompt_ms'),sha=__import__('hashlib').sha256(j.get('content','').encode()).hexdigest()[:16])
def main():
    out,order=sys.argv[1],sys.argv[2].split(','); seq=[(ci,pi,a) for ci,c in enumerate(order) for pi,a in enumerate(c)]
    groups=[]
    for x in seq:
        if groups and groups[-1][0]==x[2]: groups[-1][1].append(x)
        else: groups.append((x[2],[x]))
    log=open(out+'/units.jsonl','a')
    for arm,units in groups:
        for _ in range(180):
            if not apps(): break
            time.sleep(1)
        if apps(): sys.exit('compute apps before launch')
        name='%s-%d'%(arm.lower(),int(time.time())%100000)
        rc=sh([D+'/launch-arm.sh',name,LV[arm]]); assert rc.returncode==0,rc.stderr
        try:
            t0=time.time()
            while time.time()-t0<600:
                try:
                    if call('/health',method='GET',timeout=3)[0]==200: break
                except Exception: pass
                time.sleep(2)
            else: raise RuntimeError('not ready')
            call('/completion',{'prompt':[1,2,3,4,5,6,7,8],'n_predict':8,'temperature':0,'id_slot':0})
            if not os.path.exists(D+'/slots/'+SLOTF):   # old 2-stream slot files cannot load into -np 1: prefill once (same procedure for both arms), save, restore thereafter
                st,j=call('/completion',{'prompt':PROMPT,'n_predict':1,'id_slot':0,'cache_prompt':True,'temperature':0}); assert st==200
                st,j=call('/slots/0?action=save',{'filename':SLOTF}); assert st==200,(st,j)
            unit()  # warm-up unit, not logged
            for ci,pi,_ in units:
                for k in range(NU):
                    r=unit(); r.update(arm=arm,cycle=ci,pos=pi,k=k,container='e9-'+name); log.write(json.dumps(r)+'\n'); log.flush()
        finally:
            open(out+'/server-%s-%s.log'%(arm,name),'w').write((lambda r:r.stdout+r.stderr)(sh(['docker','logs','e9-'+name]))[-8000:])
            sh(['docker','rm','-f','e9-'+name])
if __name__=='__main__': main()
