import csv,sys,re,collections
def load(p):
    rows=list(csv.reader(open(p)))
    h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
    K=[]
    for r in rows[h+2:]:
        if len(r)!=len(hd) or r[ci["Name"]].startswith("["): continue
        K.append((float(r[ci["Start"]]),float(r[ci["Duration"]])*1000,re.sub(r'\(.*','',r[ci["Name"]].replace("void ",""))[:80]))
    heads=[i for i,k in enumerate(K) if 'head_rowpack' in k[2]]
    tok=K[heads[-2]+1:heads[-1]+1]
    ag=collections.OrderedDict()
    for t,d,n in tok:
        e=ag.setdefault(n,[0,0.0]); e[0]+=1; e[1]+=d
    return ag,sum(k[1] for k in tok)
out={}
for d in ("0","16384"): out[d]=load(f"{sys.argv[1]}/d{d}.csv")
names=sorted(set(out["0"][0])|set(out["16384"][0]),key=lambda n:-out["16384"][0].get(n,[0,0])[1]-out["0"][0].get(n,[0,0])[1])
print(f"busy ms/token: d0 {out['0'][1]/1000:.3f}  d16k {out['16384'][1]/1000:.3f}\n")
print("| kernel | n d0 | ms/tok d0 | share d0 | n 16k | ms/tok 16k | share 16k |\n|---|---|---|---|---|---|---|")
for n in names[:16]:
    a=out["0"][0].get(n,[0,0]);b=out["16384"][0].get(n,[0,0])
    print(f"| {n} | {a[0]} | {a[1]/1000:.3f} | {a[1]/out['0'][1]*100:.1f}% | {b[0]} | {b[1]/1000:.3f} | {b[1]/out['16384'][1]*100:.1f}% |")
