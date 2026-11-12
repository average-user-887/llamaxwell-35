import csv,sys,re
rows=list(csv.reader(open(sys.argv[1])))
h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
K=[]
for r in rows[h+2:]:
    if len(r)!=len(hd) or r[ci["Name"]].startswith("["): continue
    K.append((float(r[ci["Duration"]])*1000, r[ci["Name"]]))
heads=[i for i,k in enumerate(K) if 'head_rowpack' in k[1]]
# per token: kernels between consecutive head kernels (decode tokens after the first)
mv=[];bz=[]
for a,b in zip(heads[:-1],heads[1:]):
    tok=K[a+1:b+1]; bz.append(sum(k[0] for k in tok)); mv.append(sum(k[0] for k in tok if 'moe_v2' in k[1]))
n=len(mv); print(f"tokens={n} busy_us/tok={sum(bz)/n:.1f} moe_v2_us/tok={sum(mv)/n:.1f} moe_v2_min={min(mv):.1f} busy_min={min(bz):.1f}")
