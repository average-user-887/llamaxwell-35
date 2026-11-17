import json,sys,re,math,os
run=sys.argv[1]
def parse(arm):
    d=os.path.join(run,arm); R=json.load(open(d+'/result.json')); L=open(d+'/dump.txt').read().split('\n')
    res={}  # req name -> {abs_pos:(kind,top8dict)}
    for q in R['reqs']:
        a,b=q['dump_lines']; pos={}
        for ln in L[a:b]:
            m=re.match(r'([NV]) slot=\d+ n_dec=(\d+)',ln)
            if not m: continue
            k=int(m.group(2))+q['prefix_len']; rows=re.findall(r'\[([^\]]*)\]',ln)
            rows=[dict((int(x.split('=')[0]),float(x.split('=')[1])) for x in r.split(',')) for r in rows if '=' in r]
            if m.group(1)=='N': pos[k]=('N',rows[0])
            else:
                pos[k]=('V0',rows[0])
                acc=re.search(r'acc_n=(\d+)',ln)
                if len(rows)>1 and acc and int(acc.group(1))>=1: pos[k+1]=('V1',rows[1])
        res[q['name']]=pos; res[q['name']+'_tok']=R['reqs'][[x['name'] for x in R['reqs']].index(q['name'])]['tokens']
    return res
def kl(pa,pb):
    # KL(a||b) with tail lumped; top8 dicts of logprobs over union ids; ids missing in one -> use min of that dict minus tiny (bound)
    ids=set(pa)|set(pb); fa=min(pa.values()); fb=min(pb.values())
    A={i:pa.get(i,fa-0.0) for i in ids}; B={i:pb.get(i,fb) for i in ids}
    # for missing ids use that row's 8th logprob (upper bound on mass) ; renormalise not needed
    sa=sum(math.exp(v) for v in pa.values()); sb=sum(math.exp(v) for v in pb.values())
    k=0.0
    for i in pa:
        k+=math.exp(pa[i])*(pa[i]-B[i])
    ta=max(1-sa,1e-12); tb=max(1-sb,1e-12); k+=ta*math.log(ta/tb)
    return k
A=parse('A'); arms=[a for a in sorted(os.listdir(run)) if os.path.isdir(os.path.join(run,a)) and a!='A']
refpos={}
for n in ('free','tf16','tf17'):
    for p,(kd,t) in A.get(n,{}).items(): refpos.setdefault(p,t)
print('arm,req,pos,kind,A_top1,V_top1,agree,A_margin,V_margin,KL_A_V,dlp_ref,lpA_ref,lpV_ref')
rows=[]
for arm in arms:
    try: V=parse(arm)
    except Exception as e: print('#',arm,'ERR',e); continue
    for n in ('free','tf16','tf17'):
        for p,(kd,t) in sorted(V.get(n,{}).items()):
            if p not in refpos: continue
            ta=refpos[p]; sa=sorted(ta.items(),key=lambda x:-x[1]); sv=sorted(t.items(),key=lambda x:-x[1])
            r=sa[0][0]; 
            print(','.join(map(str,[arm,n,p,kd,sa[0][0],sv[0][0],int(sa[0][0]==sv[0][0]),round(sa[0][1]-sa[1][1],4),round(sv[0][1]-sv[1][1],4),round(kl(ta,t),6),round(t[r]-ta[r],5) if r in t else 'NA',round(ta[r],5),round(t.get(r,float('nan')),5)])))
    print('#',arm,'free tokens first diff vs A:',next((i for i,(x,y) in enumerate(zip(A['free_tok'],V['free_tok'])) if x!=y),None))
