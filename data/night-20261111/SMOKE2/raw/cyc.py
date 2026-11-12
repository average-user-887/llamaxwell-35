import sys,json,time
sys.argv=[sys.argv[0],sys.argv[1],sys.argv[2]]
exec(open("/media/avg-usr/Storage/m6000-llama-cpp/occamy35-night-20261111/rounds/SMOKE2/raw/drive.py").read().split("# a: concurrent")[0])
pa,pb=prompt(11,1500),prompt(22,1800)
def stream_abort(p,after):
    c=http.client.HTTPConnection("127.0.0.1",port,timeout=600)
    b=req(p,2000); b["stream"]=True
    c.request("POST","/completion",json.dumps(b),{"Content-Type":"application/json"})
    r=c.getresponse(); n=0
    while n<after:
        if r.readline().startswith(b"data:"): n+=1
    c.close()
n=3
out=[]
for cy in range(n):
    ab=[threading.Thread(target=stream_abort,args=(prompt(44+cy,1200),20)),threading.Thread(target=stream_abort,args=(prompt(55+cy,1200),40))]
    [t.start() for t in ab]; [t.join() for t in ab]
    time.sleep(1)
    o={};sub={};end={}
    def w(k,p):
        sub[k]=time.time(); o[k]=post(req(p)); end[k]=time.time()
    th=[threading.Thread(target=w,args=(k,p)) for k,p in (("s1",pa),("s2",pb))]
    [t.start() for t in th]; [t.join() for t in th]
    t0=min(sub.values()); r={}
    for k in o:
        t=o[k]["timings"]; ds=end[k]-t["predicted_ms"]/1000
        r[k]={"submit":round(sub[k]-t0,3),"prompt_start":round(ds-t["prompt_ms"]/1000-t0,3),"decode_start":round(ds-t0,3),"decode_end":round(end[k]-t0,3),"tps":round(t["predicted_per_second"],2),"ptok":t["prompt_n"]}
    out.append(r); print(cy,json.dumps(r),flush=True)
json.dump(out,open(sys.argv[2]+".json","w"))
