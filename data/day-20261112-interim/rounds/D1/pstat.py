import csv,sys,re,statistics
# pstat.py <csv...>: per token window (head_rowpack..head_rowpack): gdn_fused call #0 (pilot layer) us, all-call sum, period
for p in sys.argv[1:]:
    rows=list(csv.reader(open(p))); h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
    fd={'s':1e6,'ms':1e3,'us':1,'ns':1e-3}[rows[h+1][ci["Duration"]]]; fs={'s':1e3,'ms':1,'us':1e-3,'ns':1e-6}[rows[h+1][ci["Start"]]]
    K=[(float(r[ci["Start"]])*fs,float(r[ci["Duration"]])*fd,r[ci["Name"]],r[ci["Grid X"]]) for r in rows[h+2:] if len(r)==len(hd) and not r[ci["Name"]].startswith("[")]
    hs=[i for i,k in enumerate(K) if 'head_rowpack' in k[2]]
    l0=[];tot=[];per=[];names=set()
    for j in range(1,len(hs)-1):
        g=[k for k in K[hs[j]+1:hs[j+1]+1] if 'gdn_fused' in k[2]]
        if len(g)!=30: continue
        l0.append(g[0][1]); tot.append(sum(k[1] for k in g)); per.append(K[hs[j+1]][0]-K[hs[j]][0]); names.add(re.sub(r'\(.*','',g[0][2].replace('void ',''))+' grid='+g[0][3])
    m=statistics.median; q=lambda v: sorted(v)[len(v)//4]
    print(p.split('/')[-1],'tokens',len(l0),'L0 us median %.2f mean %.2f p25 %.2f'%(m(l0),statistics.mean(l0),q(l0)),'| gdn sum/tok median %.1f | rest29 median %.1f | period ms median %.4f mean %.4f'%(m(tot),m([a-b for a,b in zip(tot,l0)]),m(per),statistics.mean(per)),'|',names)
