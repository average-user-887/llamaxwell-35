import json,sys
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open("/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/SMOKE2/raw/drive.py").read().split("# a: concurrent")[0])
pa,pb=prompt(11,1500),prompt(22,1800); o={}
def w(k,p): b=req(p,200); b["n_probs"]=2; o[k]=post(b)
th=[threading.Thread(target=w,args=(k,p)) for k,p in (("a1",pa),("a2",pb))]
[t.start() for t in th]; [t.join() for t in th]
json.dump({k:{"text":v["content"],"probs":v["completion_probabilities"]} for k,v in o.items()},open(out,"w"))
print("ok")
