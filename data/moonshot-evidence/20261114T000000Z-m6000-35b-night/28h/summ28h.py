import sys,re,os,glob,statistics as st
out,res=sys.argv[1:3]
def val(t,k):
    m=re.search(r'^\| '+re.escape(k)+r' \| ([\d.]+)',t,re.M); return float(m.group(1)) if m else None
steps=open(out+'/steps.log').read()
CFG={'A':'batch8 libs, batch8 env (ref)','B':'28h libs 54efa6d + GGML_CUDA_GDN2_A2ROW=1'}
rows={a:[] for a in CFG}
for d in sorted(glob.glob(out+'/[AB]_*'),key=lambda p:int(p.rsplit('_',1)[1])):
    n=os.path.basename(d); a=n[0]
    try: t=open(d+'/N3_TABLE.md').read()
    except Exception: rows[a].append((n,None,None,None,False,'no table')); continue
    e=re.search(re.escape(n)+r' engaged a2row_lines=(\d+) n2gdn_launches=(\d+) moe_v3=(\d+)',steps)
    v,l,m=map(int,e.groups()) if e else (0,0,0)
    ok=l>0 and m>0 and ((v>0) if a=='B' else (v==0))
    rows[a].append((n,val(t,'makespan total s'),val(t,'max token gap s'),val(t,'peak VRAM MiB'),ok,'a2row=%d n2gdn=%d moe_v3=%d'%(v,l,m)))
def mean(a,i):
    v=[r[i] for r in rows[a] if r[4] and r[i] is not None]; return (st.mean(v),len(v),max(v)) if v else (None,0,None)
f=lambda x:'n/a' if x is None else '%.2f'%x
A=mean('A',1)
L=['## N3 ABBA short row (provisional, 2 slots)','','| arm | config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |','|---|---|---|---|---|---|---|']
for a in CFG:
    M_=mean(a,1);G=mean(a,2);V=mean(a,3)
    dl=(M_[0]-A[0])/A[0]*100 if M_[0] and A[0] else None
    L.append('| %s | %s | %d | %s | %s | %s / %s | %s |'%(a,CFG[a],M_[1],f(M_[0]),'-' if a=='A' else (f(dl)+'%' if dl is not None else 'n/a'),f(G[0]),f(G[2]),f(V[0])))
L+=['','| run | makespan s | max gap s | VRAM MiB | engaged | valid |','|---|---|---|---|---|---|']
for a in CFG:
    for r in rows[a]: L.append('| %s | %s | %s | %s | %s | %s |'%(r[0],f(r[1]),f(r[2]),f(r[3]),r[5],'yes' if r[4] else 'VOID'))
L+=['','Method: n3_replay.py short profile (card 26 pattern), prod arm, --no-sweep --no-ctxfull, order A B B A, cap1024 ub2048 b2048,',
    'env from run-35b-batch8-20261115.sh + LLAMA_PREFILL_SINGLE_STREAM=1 + GGML_CUDA_OCCAMY_DEBUG=1; MAXWELL_GRAPHS unset. A run is void unless',
    'n2gdn launches and moe_v3 launches are > 0 and the "n2gdn variant A2ROW" line is present (B) / absent (A). Noise ~0.5%. Provisional.']
open(res,'w').write('\n'.join(L)+'\n')
