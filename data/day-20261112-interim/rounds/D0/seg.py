import csv,sys,re,collections,statistics
p=sys.argv[1]
rows=list(csv.reader(open(p))); h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
u=rows[h+1]; tf={'s':1e3,'ms':1,'us':1e-3,'ns':1e-6}[u[ci['Start']]]; df={'s':1e6,'ms':1e3,'us':1,'ns':1e-3}[u[ci['Duration']]]
K=[]
for r in rows[h+2:]:
    if len(r)!=len(hd) or r[ci['Name']].startswith('['): continue
    K.append((float(r[ci['Start']])*tf,float(r[ci['Duration']])*df,re.sub(r'\(.*','',r[ci['Name']].replace('void ',''))[:70]))
hs=[i for i,k in enumerate(K) if 'head_rowpack' in k[2]]
# runs: consecutive heads with gap<60ms
runs=[];cur=[hs[0]]
for a,b in zip(hs,hs[1:]):
    if K[b][0]-K[a][0]<60: cur.append(b)
    else: runs.append(cur);cur=[b]
runs.append(cur)
print('runs',[len(r) for r in runs])
names=['16k','64k','96k','2x deep(96k+90k)']
big=[r for r in runs if len(r)>=15]
print('big runs',[len(r) for r in big])
for nm,r in zip(names,big):
    r=r[2:] if len(r)>8 else r   # drop first 2 tokens (first-token/prefill tail)
    per=[K[r[j+1]][0]-K[r[j]][0] for j in range(len(r)-1)]
    toks=[K[r[j]+1:r[j+1]+1] for j in range(len(r)-1)]
    busy=[sum(d for _,d,_ in t)/1000 for t in toks]
    ps=sorted(per);tr=ps[1:-1]
    ag=collections.defaultdict(float);cn=collections.defaultdict(int)
    for t in toks:
        for _,d,n in t: ag[n]+=d/len(toks);cn[n]+=1/len(toks)
    print('\n== %s: %d tokens; period ms median %.3f trimmed %.3f; busy ms/tok mean %.3f (median %.3f) busy/period %.3f'%(nm,len(per),statistics.median(per),sum(tr)/len(tr),statistics.mean(busy),statistics.median(busy),statistics.mean(busy)/statistics.mean(per)))
    for n,v in sorted(ag.items(),key=lambda x:-x[1])[:10]: print('   %8.3f ms/tok %5.1f%%  %5.1f calls/tok  %s'%(v/1000,100*v/1000/statistics.mean(busy),cn[n],n))
    for n in ag:
        if 'ofa_' in n and n not in [x for x,_ in sorted(ag.items(),key=lambda x:-x[1])[:10]]: print('   %8.3f ms/tok (ofa) %s'%(ag[n]/1000,n))
