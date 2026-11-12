import csv,sys,statistics as S
for f in sys.argv[1:]:
    rows=[l for l in open(f) if not l.startswith('==')]
    r=list(csv.reader(rows)); h=r[0]; r=r[2:]
    i=h.index('Start'); d=h.index('Duration'); n=h.index('Name'); sz=h.index('Size')
    ev=[(float(x[i]),float(x[d] or 0)*1e-3,x[n],x[sz]) for x in r if len(x)>n]
    gaps=[];per=[]
    idx=[k for k,e in enumerate(ev) if 'DtoH' in e[2] and e[3] and float(e[3])>0.5]
    for k in idx:
        end=ev[k][0]+ev[k][1]
        nk=next((e for e in ev[k+1:] if 'CUDA memcpy' not in e[2] and 'hg_gather' not in e[2]),None)
        if nk: gaps.append((nk[0]-end)*1e6)
    ts=[ev[k][0] for k in idx]; per=[(b-a)*1e3 for a,b in zip(ts,ts[1:])][2:]
    g=gaps[2:-1]
    print(f.split('/')[-1],'dtoh_end->first_compute_kernel us median %.1f mean %.1f'%(S.median(g),S.mean(g)),'| period ms trimmed-mean %.4f'%S.mean(sorted(per)[2:-2]))
