import csv,sys,re,collections,statistics
# ktable.py <csv>: per-kernel table over decode token windows (head_rowpack to head_rowpack, as busy.py/period.py), + resources
p=sys.argv[1]
rows=list(csv.reader(open(p))); h=[i for i,r in enumerate(rows) if r and r[0]=="Start"][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
u=rows[h+1]; tf={'s':1e3,'ms':1,'us':1e-3,'ns':1e-6}[u[ci['Start']]]; df={'s':1e6,'ms':1e3,'us':1,'ns':1e-3}[u[ci['Duration']]]
K=[]
for r in rows[h+2:]:
    if len(r)!=len(hd) or r[ci['Name']].startswith('['): continue
    K.append(dict(s=float(r[ci['Start']])*tf,d=float(r[ci['Duration']])*df,n=r[ci['Name']],g='x'.join(r[ci[c]] for c in ('Grid X','Grid Y','Grid Z')),b='x'.join(r[ci[c]] for c in ('Block X','Block Y','Block Z')),reg=r[ci['Registers Per Thread']],ss=r[ci['Static SMem']],ds=r[ci['Dynamic SMem']]))
hs=[i for i,k in enumerate(K) if 'head_rowpack' in k['n']]
per=[(K[hs[j+1]]['s']-K[hs[j]]['s']) for j in range(1,len(hs)-2)]
toks=[K[hs[j]+1:hs[j+1]+1] for j in range(1,len(hs)-1)]
busy=[sum(k['d'] for k in t)/1000 for t in toks]
ps=sorted(per); tr=ps[1:-1]
print('tokens',len(toks),'period ms median %.4f trimmed %.4f mean %.4f'%(statistics.median(per),sum(tr)/len(tr),statistics.mean(per)))
print('busy ms/tok median %.4f mean %.4f'%(statistics.median(busy),statistics.mean(busy)),'busy/period %.3f'%(statistics.mean(busy)/statistics.mean(per)))
ag=collections.OrderedDict()
for t in toks:
    for k in t:
        key=(re.sub(r'\(.*','',k['n'].replace('void ',''))[:110],k['g'],k['b'],k['reg'],k['ss'],k['ds'])
        ag.setdefault(key,[]).append(k['d'])
tot=sum(sum(v) for v in ag.values()); nt=len(toks)
print('total busy us',round(tot),'per token us',round(tot/nt,1))
print('%busy | calls/tok | total_us/tok | mean_us | grid | block | reg | sSM KB | dSM KB | kernel')
for key,v in sorted(ag.items(),key=lambda x:-sum(x[1])):
    print('%5.2f%% %6.2f %8.1f %7.2f %s %s %s %s %s %s'%(100*sum(v)/tot,len(v)/nt,sum(v)/nt,statistics.mean(v),key[1],key[2],key[3],key[4],key[5],key[0]))
