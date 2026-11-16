import sys,re,collections,subprocess,json,glob,os
R=sys.argv[1]
def dem(n):
    try: return subprocess.run(['c++filt',n],capture_output=True,text=True).stdout.strip()
    except Exception: return n
def load(p):
    names={};lab={};segs={};K=collections.defaultdict(list)
    for l in open(p,errors='replace'):
        f=l.rstrip('\n').split('\t')
        if f[0]=='L': lab[int(f[1])]=f[2]
        elif f[0]=='N': names[f[1]]=f[2]
        elif f[0]=='S': segs[int(f[1])]=dict(label=lab.get(int(f[2]),'?'),nl=int(f[4]),gpu=float(f[5]),t0=float(f[6]),t1=float(f[7]),mb=int(f[8]),mms=float(f[9]))
        elif f[0]=='K': K[int(f[1])].append((f[2],int(f[3]),float(f[4])))
    return names,segs,K
def comp(n):
    s=n
    if re.search(r'gated_delta|gdn|delta_net|ssm_conv|l2_norm',s,re.I): return 'GDN'
    if re.search(r'flash_attn|fattn|occamy_fa|fa_gemm|maxwell_gemm|kq_|soft_max|rope',s,re.I): return 'attn'
    if re.search(r'moe|mul_mat_id|mmid|topk|argsort|mmq_id|ids',s,re.I): return 'MoE'
    if re.search(r'mul_mat|mmvq|mmq|gemm|cublas|sgemm|gemv',s,re.I): return 'dense'
    return 'other'
for arm in sorted(os.listdir(R)):
    p=f'{R}/{arm}/prof.txt'
    if not os.path.exists(p) or os.path.getsize(p)<100: continue
    names,segs,K=load(p)
    print('=====',arm)
    byl=collections.defaultdict(list)
    for i,s in segs.items(): byl[s['label']].append(i)
    for l,ids in byl.items():
        g=collections.defaultdict(list)
        for i in ids: g[segs[i]['nl']//20*20 if segs[i]['nl']>0 else 0].append(i)
        print(' label',l,'nsegs',len(ids))
        for nl,ii in sorted(g.items(),key=lambda x:-len(x[1])):
            if len(ii)<5 or nl==0: continue
            print('   nl~%d n=%d gpu_ms=%.2f'%(nl,len(ii),sum(segs[i]['gpu'] for i in ii)/len(ii)))

print('\n######## DETAIL')
LAB={'np1':{1:'1s_d80K',2:'1s_d90K'},'np2':{1:'1s_d80K',2:'2s_d80K'}}
rows=[];offk={}
detail={}
for arm in sorted(os.listdir(R)):
    p=f'{R}/{arm}/prof.txt'
    if not os.path.exists(p) or os.path.getsize(p)<100: continue
    names,segs,K=load(p); npk='np2' if arm.endswith('np2') else 'np1'
    # raw label ids
    ids={}
    for l in open(p,errors='replace'):
        f=l.split('\t')
        if f[0]=='S': ids.setdefault(int(f[2]),[]).append(int(f[1]))
    for lid,sl in sorted(ids.items()):
        sl=sorted(sl); lb=segs[sl[0]]['label']
        mx=max(segs[i]['nl'] for i in sl)
        tg=[i for i in sl if segs[i]['nl']>=0.6*mx]; ot=[i for i in sl if 0<segs[i]['nl']<0.6*mx]
        # for 2 slot spec, also separate target by nl clusters is not needed
        n=len(tg)
        c=collections.defaultdict(float);cnt=collections.defaultdict(float);kern=collections.defaultdict(lambda:[0,0.0])
        for i in tg:
            for (pt,k,ms) in K[i]:
                nm=dem(names.get(pt,pt)); c[comp(nm)]+=ms; kern[nm][0]+=k; kern[nm][1]+=ms
            c['d2h_marker']+=segs[i]['mms']
        mtp=sum(segs[i]['gpu'] for i in ot)/n
        tgt=sum(segs[i]['gpu'] for i in tg)/n
        t1=[segs[i]['t1'] for i in sorted(tg)]
        wall=(t1[-1]-t1[0])/(n-1)*1000 if n>1 else 0
        rows.append((arm,lb,n,len(ot)/n,mx,tgt,{k:v/n for k,v in c.items()},mtp,wall,wall-tgt-mtp))
        detail[(arm,lb)]={k:(v[0]/n,v[1]/n) for k,v in kern.items()}
print('| arm | label | steps | draftsegs/step | nl/step | target GPU ms | attn | MoE | GDN | dense | other | D2H | MTP-head ms/step | wall ms/step | host gap ms |')
print('|'+'---|'*15)
for (arm,lb,n,ds,mx,tgt,c,mtp,wall,gap) in rows:
    print('| %s | %s | %d | %.2f | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f | %.1f | %.1f | %.1f |'%(arm,lb,n,ds,mx,tgt,c.get('attn',0),c.get('MoE',0),c.get('GDN',0),c.get('dense',0),c.get('other',0),c.get('d2h_marker',0),mtp,wall,gap))
json.dump({'%s|%s'%k:v for k,v in detail.items()},open(R+'/kern_detail.json','w'))
offset=set(detail[('off_np1','s1_d80000')])
for k in [('n1_np1','s1_d80000'),('n2_np1','s1_d80000')]:
    d=detail.get(k)
    if not d: continue
    print('\n## top kernels',k,'(per step count, ms; NEW=not in off 1-row set)')
    for nm,(c_,ms) in sorted(d.items(),key=lambda x:-x[1][1])[:22]: print('  %6.1f x %7.2f ms %s %s'%(c_,ms,'NEW' if nm not in offset else '   ',nm[:110]))
    print('  NEW kernels total:',sum(1 for nm in d if nm not in offset),'count/step %.0f ms/step %.1f'%(sum(v[0] for nm,v in d.items() if nm not in offset),sum(v[1] for nm,v in d.items() if nm not in offset)))
    for nm,(c_,ms) in sorted(d.items(),key=lambda x:-x[1][1]):
        if nm not in offset: print('   NEW %6.1f x %6.2f ms %s'%(c_,ms,nm[:140]))
