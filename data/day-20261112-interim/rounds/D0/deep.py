import json,sys,time,threading,urllib.request,subprocess
U='http://127.0.0.1:18490'
def post(p,d,t=3600):
    r=urllib.request.Request(U+p,json.dumps(d).encode(),{'Content-Type':'application/json'}); return json.load(urllib.request.urlopen(r,timeout=t))
def wait():
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(U+'/health',timeout=3)).get('status')=='ok': return
        except Exception: pass
        time.sleep(2)
    sys.exit('server not ready')
def toks():
    t=open('/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D0/deep_text.txt').read()
    ids=[]
    for i in range(0,len(t),400000): ids+=post('/tokenize',{'content':t[i:i+400000]})['tokens']
    return ids
def comp(ids,n,slot,out,key):
    r=post('/completion',{'prompt':ids,'n_predict':n,'id_slot':slot,'cache_prompt':True,'temperature':0,'ignore_eos':True})
    out[key]=r['timings']
if __name__=='__main__':
    mode=sys.argv[1]; wait(); ids=toks(); print('tokens avail',len(ids),flush=True)
    res={}
    def run(n_a,off_a,n_b=None,off_b=None,n=1,tag=''):
        th=[];o={}
        a=threading.Thread(target=comp,args=(ids[off_a:off_a+n_a],n,0,o,'s0')); th.append(a)
        if n_b: th.append(threading.Thread(target=comp,args=(ids[off_b:off_b+n_b],n,1,o,'s1')))
        t0=time.time(); [x.start() for x in th]; [x.join() for x in th]; res[tag]={'wall':time.time()-t0,**{k:{kk:v[kk] for kk in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_per_second')} for k,v in o.items()}}; print(tag,json.dumps(res[tag]),flush=True)
    if mode=='prefill':  # f + VRAM
        run(65536,0,tag='slot0_64k_prefill',n=16)
        run(98000,100000,tag='slot0_98k_prefill',n=16)
        run(90000,0,90000,100000,tag='both_90k_prefill_concurrent',n=1)  # slot0 90K (new prefix), slot1 90K
        run(90000,0,90000,100000,tag='both_90k_decode_cached',n=48)
    elif mode=='prof2':
        run(90000,0,90000,100000,tag='both_prefill',n=1)
        run(90000,0,90000,100000,tag='both_decode_cached',n=48)
    json.dump(res,open('/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D0/deep_%s.json'%mode,'w'),indent=1)
