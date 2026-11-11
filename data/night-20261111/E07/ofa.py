import csv,sys,re,collections
rows=list(csv.reader(open(sys.argv[1])))
h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
unit=rows[h+1][ci["Duration"]]; f={'s':1e6,'ms':1e3,'us':1,'ns':1e-3}[unit]
ag=collections.defaultdict(list)
for r in rows[h+2:]:
    if len(r)!=len(hd): continue
    n=r[ci["Name"]]
    if 'ofa_' in n:
        key=re.sub(r'\(.*','',n.replace('void ',''))+' grid='+'x'.join(r[ci[c]] for c in ('Grid X','Grid Y','Grid Z'))+' regs='+r[ci["Registers Per Thread"]]
        ag[key].append(float(r[ci["Duration"]])*f)
for k,v in ag.items():
    t=sorted(v[-50:]); print(f"{k}: n={len(v)} last50 median={t[len(t)//2]:.1f}us mean={sum(t)/len(t):.1f}us")
