import json,sys,glob,os,collections
O=sys.argv[1]; arms=['offA','offA2','mtp2','mtp3','mtp1','ngmod','ngmapk','ngmod_mtp2']
def L(a):
    p=glob.glob('%s/%s/run/run.json'%(O,a)); return json.load(open(p[0])) if p else None
R={a:L(a) for a in arms}
def key(r): return {(t['agent'],t['turn']):t for t in r['turns_rec']}
base=key(R['offA']) if R['offA'] else {}
def eq(a,b):
    if not R.get(a) or not R.get(b): return 'n/a'
    ka,kb=key(R[a]),key(R[b]); d=sum(1 for k in set(ka)|set(kb) if ka.get(k,{}).get('token_sha')!=kb.get(k,{}).get('token_sha')); return '%d/%d differ'%(d,len(set(ka)|set(kb)))
def peak(a):
    try: return max(int(x) for x in open('%s/%s/vram.csv'%(O,a)).read().split())
    except Exception: return None
print('| arm | kind | drafted | accepted | acc/gen | zero-accept turns | decode s | makespan s | VRAM MiB | sha vs offA | sha vs offA2 | void |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in arms:
    r=R[a]
    if not r: print('| %s | not run |||||||||||'%a); continue
    spec=a not in('offA','offA2'); bys=collections.defaultdict(list)
    for t in r['turns_rec']: bys[t['kind']].append(t)
    sl=collections.defaultdict(int)
    for t in r['turns_rec']: sl[t['agent']]+=t.get('draft_n') or 0
    void='VOID' if spec and not all(sl[s]>0 for s in (0,1) if True) and len(sl)>=2 and min(sl.values())==0 or (spec and len(sl)<2) else 'ok'
    for k,ts in list(bys.items())+[('ALL',r['turns_rec'])]:
        d=sum(t.get('draft_n') or 0 for t in ts); ac=sum(t.get('draft_n_accepted') or 0 for t in ts); g=sum(t.get('tokens_predicted') or 0 for t in ts)
        z=sum(1 for t in ts if not t.get('draft_n_accepted')); ds=sum((t.get('predicted_ms') or 0) for t in ts)/1000
        last=k=='ALL'
        print('| %s | %s | %d | %d | %.3f | %d/%d | %.1f | %s | %s | %s | %s | %s |'%(a,k,d,ac,ac/g if g else 0,z,len(ts),ds,'%.1f'%r['makespan_s'] if last else '',peak(a) if last else '',eq('offA',a) if last and a!='offA' else '',eq('offA2',a) if last and a!='offA2' else '',void if last else ''))
