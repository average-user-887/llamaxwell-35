import csv,sys,re,collections,statistics
# kstat.py <csv...>: gdn_fused per-call us within decode token windows (head_rowpack to head_rowpack, same windows as busy.py)
for p in sys.argv[1:]:
    rows=list(csv.reader(open(p))); h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
    f={'s':1e6,'ms':1e3,'us':1,'ns':1e-3}[rows[h+1][ci["Duration"]]]
    K=[(float(r[ci["Duration"]])*f,r[ci["Name"]],'x'.join(r[ci[c]] for c in ('Grid X','Grid Y','Grid Z'))) for r in rows[h+2:] if len(r)==len(hd) and not r[ci["Name"]].startswith("[")]
    hs=[i for i,k in enumerate(K) if 'head_rowpack' in k[1]]
    ag=collections.defaultdict(list); per=[]
    for j in range(1,len(hs)-1):
        s=0
        for d,n,g in K[hs[j]+1:hs[j+1]+1]:
            if 'gdn_fused' in n: ag[re.sub(r'\(.*','',n.replace('void ',''))+' grid='+g].append(d); s+=d
        per.append(s)
    allc=[d for v in ag.values() for d in v]
    print(p.split('/')[-1],'gdn calls',len(allc),'median %.2f mean %.2f us/call'%(statistics.median(allc),statistics.mean(allc)),'sum/tok %.1f us'%statistics.mean(per))
    for k,v in ag.items(): print('  ',k,'n',len(v),'median %.2f mean %.2f'%(statistics.median(v),statistics.mean(v)))
