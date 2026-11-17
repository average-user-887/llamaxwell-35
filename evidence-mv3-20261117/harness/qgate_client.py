# usage: qgate_client.py HOST:PORT out.json dumpfile base_result.json
# req0: 80K cold, greedy 48 tok. req1/req2: teacher-forced prefix base[:16] / base[:17], 8 tok (fresh-state window around idx 18).
import json,sys,glob,os,urllib.request as U
hp,out,dump,refp=sys.argv[1:5]; B='http://'+hp
def post(p,b):
    with U.urlopen(U.Request(B+p,json.dumps(b).encode(),{'Content-Type':'application/json'}),timeout=7200) as f: return json.loads(f.read())
txt=''.join(open(f,errors='ignore').read() for f in sorted(glob.glob('/usr/lib/python3.12/*.py')))
P=post('/tokenize',{'content':txt[:80000*6+60000]})['tokens'][:80000]
base=json.load(open(refp))['runs']['cold']['slots']['0']['out_tokens']
R={'reqs':[]}
def nl(): return sum(1 for _ in open(dump)) if os.path.exists(dump) else 0
def go(name,prompt,n):
    a=nl()
    r=post('/completion',{'prompt':prompt,'n_predict':n,'temperature':0,'ignore_eos':True,'cache_prompt':True,'id_slot':0,'return_tokens':True})
    R['reqs'].append({'name':name,'prefix_len':len(prompt)-len(P),'dump_lines':[a,nl()],'tokens':r.get('tokens'),'timings':{k:r['timings'].get(k) for k in ('prompt_n','predicted_n','draft_n','draft_n_accepted')}})
    json.dump(R,open(out,'w'),indent=1)
go('free',P,48)
go('tf16',P+base[:16],8)
go('tf17',P+base[:17],8)
R['done']=True; json.dump(R,open(out,'w'),indent=1)
