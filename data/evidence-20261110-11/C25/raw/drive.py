import json, random, sys, time, threading, urllib.request, http.client, hashlib
port=int(sys.argv[1]); out=sys.argv[2]
S=["The harbor authority recorded that vessel %d arrived with %d containers of grain.","In the valley, farmer %d counted %d sheep before the storm reached the ridge.","Engineer %d reported that pump %d lost pressure after the third inspection.","The archive lists document %d as referencing ledger %d and a missing seal.","A courier carried parcel %d through checkpoint %d without further delay.","Station %d logged a temperature drift of %d millikelvin overnight.","Student %d solved exercise %d by induction and a careful case split.","The orchestra rehearsed movement %d in hall %d until the oboe tuned.","Cartographer %d revised sheet %d after the river changed its course.","Clerk %d filed form %d in triplicate and stamped each copy."]
def prompt(seed, ntok):
    r=random.Random(seed); t=[]
    while len(" ".join(t))<ntok*4.3: t.append(r.choice(S)%(r.randint(1,999),r.randint(1,999)))
    return " ".join(t)+"\n\nQuestion: Write a short summary of the above, listing three notable numbers.\nAnswer:"
def post(body, timeout=900):
    rq=urllib.request.Request(f"http://127.0.0.1:{port}/completion",json.dumps(body).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(rq,timeout=timeout))
def req(p,n=200): return {"prompt":p,"n_predict":n,"temperature":0,"seed":1,"top_k":1,"cache_prompt":False}
res={}
def tm(r): t=r["timings"]; return {"prompt_n":t["prompt_n"],"prompt_tps":round(t["prompt_per_second"],2),"predicted_n":t["predicted_n"],"decode_tps":round(t["predicted_per_second"],2)}
# a: concurrent
pa,pb=prompt(11,1500),prompt(22,1800); o={}
def w(k,p): o[k]=post(req(p))
th=[threading.Thread(target=w,args=(k,p)) for k,p in (("a1",pa),("a2",pb))]
[t.start() for t in th]; [t.join() for t in th]
res["a"]={k:{"text":v["content"],"t":tm(v),"stop":v.get("stop_type")} for k,v in o.items()}
# b: long
v=post(req(prompt(33,16000),200)); res["b"]={"text":v["content"],"t":tm(v),"stop":v.get("stop_type")}
# c: cancel mid-generation, then new requests
def stream_abort(p, after=20):
    c=http.client.HTTPConnection("127.0.0.1",port,timeout=600)
    b=req(p,2000); b["stream"]=True
    c.request("POST","/completion",json.dumps(b),{"Content-Type":"application/json"})
    r=c.getresponse(); n=0
    while n<after:
        l=r.readline()
        if l.startswith(b"data:"): n+=1
    c.close(); return n
ab=[]
ca=threading.Thread(target=lambda: ab.append(stream_abort(prompt(44,1200))))
cb=threading.Thread(target=lambda: ab.append(stream_abort(prompt(55,1200),40)))
ca.start(); cb.start(); ca.join(); cb.join()
time.sleep(1)
v1=post(req(pa,200)); res["c1"]={"text":v1["content"],"t":tm(v1)}   # same as a1, single slot post-cancel
o={}
th=[threading.Thread(target=w,args=(k,p)) for k,p in (("c2a",pa),("c2b",pb))]
[t.start() for t in th]; [t.join() for t in th]
res["c2"]={k:{"text":v["content"],"t":tm(v)} for k,v in o.items()}
res["c_aborted_after"]=ab
# d: single-slot decode tps on short prompt
v=post(req(prompt(66,300),300)); res["d"]={"text":v["content"],"t":tm(v)}
json.dump(res,open(out,"w"),indent=1)
print("done",{k:(res[k]["t"] if "t" in res[k] else None) for k in ("b","d")})
