import csv,sys,statistics as S
def load(f):
    rows=[l for l in open(f) if not l.startswith('==')]
    r=list(csv.reader(rows)); h=r[0]; r=r[2:]
    i=h.index('Start'); d=h.index('Duration'); n=h.index('Name'); sz=h.index('Size') if 'Size' in h else None
    return [(float(x[i]),float(x[d]) if x[d] else 0,x[n],x[sz] if sz is not None else '') for x in r if len(x)>n]
for f in sys.argv[1:]:
    ev=load(f)
    dtoh=[e for e in ev if 'DtoH' in e[2] and e[3] and float(e[3])>0.5]  # logits ~0.9MB
    ts=[e[0] for e in dtoh]; per=[(b-a)*1e3 for a,b in zip(ts,ts[1:])]
    per=per[2:]  # skip warmup
    htod=sum(1 for e in ev if 'HtoD' in e[2])
    print(f.split('/')[-1],'tokens',len(ts),'period_ms median %.4f mean %.4f'%(S.median(per),S.mean(per)),'htod_total',htod,'gather',sum(1 for e in ev if 'hg_gather' in e[2]))
