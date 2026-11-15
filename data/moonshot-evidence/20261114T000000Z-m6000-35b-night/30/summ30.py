import sys,re,os,glob,statistics as st
out,res=sys.argv[1:3]
def val(t,k):
    m=re.search(r'^\| '+re.escape(k)+r' \| ([\d.]+)',t,re.M); return float(m.group(1)) if m else None
steps=open(out+'/steps.log').read()
CFG={'A':'ref (cap1024 ub2048 b2048, default host flags)','B':'--threads 4 --threads-batch 4 --poll 100','C':'--poll 0','D':'--prio 2 --poll 100'}
rows={a:[] for a in CFG}; SS={}
for d in sorted(glob.glob(out+'/[A-D]_*'),key=lambda p:int(p.rsplit('_',1)[1])):
    n=os.path.basename(d); a=n[0]
    try: t=open(d+'/N3_TABLE.md').read()
    except Exception: rows[a].append((n,None,None,None,False,(0,0,0))); continue
    e=re.search(re.escape(n)+r' engaged gemm=(\d+) fused=(\d+) moe_v3=(\d+)',steps)
    g,f,m=map(int,e.groups()) if e else (0,0,0)
    ss=re.search(re.escape(n)+r' ss_lines: (.*)',steps); SS[n]=ss.group(1) if ss else ''
    rows[a].append((n,val(t,'makespan total s'),val(t,'max token gap s'),val(t,'peak VRAM MiB'),g>0 and f>0 and m>0,(g,f,m)))
def mean(a,i):
    v=[r[i] for r in rows[a] if r[4] and r[i] is not None]; return (st.mean(v),len(v),max(v)) if v else (None,0,None)
f=lambda x:'n/a' if x is None else '%.2f'%x
A=mean('A',1); vA=mean('A',3)
L=['# Card 30 RESULT: host flag A/B (9/bin, GEMM+FUSED+SINGLE_STREAM+MOE_V3)','','| arm | arm config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |','|---|---|---|---|---|---|---|']
win=[]
for a in CFG:
    M_=mean(a,1);G=mean(a,2);V=mean(a,3)
    dl=(M_[0]-A[0])/A[0]*100 if M_[0] and A[0] else None
    L.append('| %s | %s | %d | %s | %s | %s / %s | %s |'%(a,CFG[a],M_[1],f(M_[0]),'-' if a=='A' else (f(dl)+'%' if dl is not None else 'n/a'),f(G[0]),f(G[2]),f(V[0])))
    if a!='A' and dl is not None and dl<=-0.5 and G[2]<=6.5: win.append('%s (%.2f%%)'%(a,dl))
L+=['','Verdict: **%s** (gate: >=0.5%% faster than A, max gap <= 6.5 s)'%('ARMS PASS: '+', '.join(win) if win else 'no arm passes'),'','## Per run','','| run | arm config | makespan s | max gap s | VRAM MiB | engaged (gemm,fused,moe_v3) | valid |','|---|---|---|---|---|---|---|']
for a in CFG:
    for r in rows[a]: L.append('| %s | %s | %s | %s | %s | %s | %s |'%(r[0],CFG[a],f(r[1]),f(r[2]),f(r[3]),r[5],'yes' if r[4] else 'VOID/failed'))
L+=['','## Per-run server.log batch lines (see steps.log)','']+['- %s: %s'%(k,v or '(none)') for k,v in SS.items()]
L+=['','## Method','Short N3 replay (n3_replay.py, prod arm, --no-sweep --no-ctxfull), 2-slot, order A B C D D C B A. All arms 9/bin, env GEMM=1 FUSED=1 PREFILL_SINGLE_STREAM=1 GGML_CUDA_MOE_V3=1 GGML_CUDA_OCCAMY_DEBUG=1, MAXWELL_GRAPHS unset. cap1024 ub2048. Host flags per table via N3_SERVER_ARGS hook (copy of harness h/n3_replay.py; appended to llama-server args). Base cap1024 ub2048 b2048. Runs lacking engagement are void. Lock: flock gpu-bench.lock; prod unloaded via POST.','','## Caveats','n_batch/n_ubatch per arm are in the steps.log n_ubatch lines. Makespan noise about 0.5%.','','## Files','Run dir: '+out+'; harness: '+os.path.dirname(res)+'/gpu_ab.sh, h/, summ30.py']
open(res,'w').write('\n'.join(L)+'\n')
