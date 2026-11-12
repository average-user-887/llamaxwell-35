import csv,sys,statistics
# token period = Start(head_rowpack[j+1]) - Start(head_rowpack[j]), ms; plus busy for same windows
def load(p):
    rows=list(csv.reader(open(p)))
    h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
    un=rows[h+1]; s_unit=un[ci["Start"]]
    k=[(float(r[ci["Start"]]),r[ci["Name"]]) for r in rows[h+2:] if len(r)==len(hd) and not r[ci["Name"]].startswith("[")]
    st=[s for s,n in k if 'head_rowpack' in n]
    sc={'s':1e3,'ms':1,'us':1e-3,'ns':1e-6}[s_unit]
    return [(b-a)*sc for a,b in zip(st[1:-1],st[2:])]
for p in sys.argv[1:]:
    P=load(p); Ps=sorted(P); t=Ps[1:-1] if len(Ps)>2 else Ps
    print(p.split('/')[-1],"periods",[round(x,3) for x in P],"median",round(statistics.median(P),4),"trim",round(sum(t)/len(t),4))
