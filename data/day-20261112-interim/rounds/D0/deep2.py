import json,sys,time,threading
sys.path.insert(0,'/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D0')
from deep import *
wait(); ids=toks(); print('tokens avail',len(ids),flush=True); res={}
def run(n_a,off_a,n_b=None,off_b=None,n=1,tag=''):
    th=[];o={}
    if n_a: th.append(threading.Thread(target=comp,args=(ids[off_a:off_a+n_a],n,0,o,'s0')))
    if n_b: th.append(threading.Thread(target=comp,args=(ids[off_b:off_b+n_b],n,1,o,'s1')))
    t0=time.time(); [x.start() for x in th]; [x.join() for x in th]
    res[tag]={'t0':t0,'wall':time.time()-t0,**{k:{kk:v[kk] for kk in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_per_second')} for k,v in o.items()}}; print(tag,json.dumps(res[tag]),flush=True)
run(16384,0,n=24,tag='s0_16k')
run(65536,0,n=24,tag='s0_64k')
run(98000,0,n=24,tag='s0_96k')
print(post('/slots/0?action=save',{'filename':'s0_98k_off0.bin'}),flush=True)
run(0,0,90000,200000,n=1,tag='s1_90k_prefill')
print(post('/slots/1?action=save',{'filename':'s1_90k_off200000.bin'}),flush=True)
run(98000,0,90000,200000,n=48,tag='both_deep_decode')
json.dump(res,open('/media/avg-usr/Storage/m6000-llama-cpp/occamy35-day-20261112/rounds/D0/deep2.json','w'),indent=1)
