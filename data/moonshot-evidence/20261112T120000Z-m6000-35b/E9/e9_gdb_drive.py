#!/usr/bin/env python3
"""restore ~98K slot, start long decode in background thread, take N gdb thread-apply-all-bt samples inside container e9-gdb. usage: e9_gdb_drive.py <out> <N>"""
import sys, os, json, time, subprocess, threading, http.client
E='<LAB>/occamy35-day-20261112/evidence'; D=E+'/E9'; out,N=sys.argv[1],int(sys.argv[2]); SLOTF='e9_np1_98k.bin'
def call(path,body,timeout=1800):
    c=http.client.HTTPConnection('127.0.0.1',18494,timeout=timeout); c.request('POST',path,json.dumps(body),{'Content-Type':'application/json'}); r=c.getresponse(); d=r.read(); c.close(); return r.status,d
d0=json.load(open(E+'/fixtures/tokens/d0.json')); P=d0[:98047]+[198]
if not os.path.exists(D+'/slots/'+SLOTF):
    call('/completion',{'prompt':P,'n_predict':1,'id_slot':0,'temperature':0}); st,_=call('/slots/0?action=save',{'filename':SLOTF}); assert st==200
st,d=call('/slots/0?action=restore',{'filename':SLOTF}); assert st==200,d
res={'n_req':0,'stop':False}
def gen():   # context window leaves only ~250 tokens above the 98K restored context, so loop restore + 240-token decodes
    while not res['stop']:
        try:
            call('/slots/0?action=restore',{'filename':SLOTF}); call('/completion',{'prompt':P,'n_predict':240,'id_slot':0,'cache_prompt':True,'temperature':0,'ignore_eos':True}); res['n_req']+=1
        except Exception as e: res['err']=repr(e); time.sleep(1)
t=threading.Thread(target=gen,daemon=True); t.start(); time.sleep(3)
pid=subprocess.run(['docker','exec','e9-gdb','pgrep','-f','llama-server'],capture_output=True,text=True).stdout.split()[0]
open(out+'/pid_in_container.txt','w').write(pid)
os.makedirs(out+'/samples',exist_ok=True); ok=0
for i in range(N):
    r=subprocess.run(['docker','exec','e9-gdb','gdb','-p',pid,'-batch','-ex','set pagination off','-ex','thread apply all bt'],capture_output=True,text=True,timeout=300)
    open(out+'/samples/s%03d.txt'%i,'w').write(r.stdout); ok+=1
    if time.time()>float(os.environ.get('DEADLINE','1e18')): print('deadline at sample',i); break
    time.sleep(0.2)
res['stop']=True; print('samples',ok,'decode requests completed',res['n_req'],res.get('err'))
