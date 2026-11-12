import csv,sys,re,collections
def load(p):
    rows=list(csv.reader(open(p)))
    h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
    K=[]
    for r in rows[h+2:]:
        if len(r)!=len(hd) or r[ci["Name"]].startswith("["): continue
        K.append((float(r[ci["Duration"]])*1000,re.sub(r'\(.*','',r[ci["Name"]].replace("void ",""))[:60]))
    heads=[i for i,k in enumerate(K) if 'head_rowpack' in k[1]]
    toks=[K[heads[j]+1:heads[j+1]+1] for j in range(1,len(heads)-1)]
    return toks
for p in sys.argv[1:]:
    toks=load(p); b=[sum(d for d,_ in t) for t in toks]
    ag=collections.Counter()
    for t in toks:
        for d,n in t:
            if 'mul_mat_vec_q<' in n or 'occamy_attn_qkv' in n: ag[n]+=d/len(toks)
    print(p.split('/')[-1], "tokens",len(toks),"busy ms/tok",[round(x/1000,4) for x in b],"mean",round(sum(b)/len(b)/1000,4),{k[:40]:round(v/1000,4) for k,v in ag.items()})
