import json,sys,statistics as S,collections
R=[json.loads(l) for l in open(sys.argv[1]+'/units.jsonl')]
print('units',len(R),'distinct output sha per arm:',{a:len({r['sha'] for r in R if r['arm']==a}) for a in 'AB'})
by=collections.defaultdict(list)
for r in R: by[(r['arm'],r['cycle'],r['pos'],r['container'])].append(r)
print('per block (arm cycle pos): n, mean ms/tok, mean wall s, mean decode wall s')
for k,v in sorted(by.items(),key=lambda x:(x[0][1],x[0][2])): print(k[0],k[1],k[2],len(v),'%.3f'%S.mean(r['ms_per_tok'] for r in v),'%.4f'%S.mean(r['wall_s'] for r in v),'%.4f'%S.mean(r['gen_wall_s'] for r in v))
for m in ('ms_per_tok','wall_s','gen_wall_s'):
    a=[r[m] for r in R if r['arm']=='A']; b=[r[m] for r in R if r['arm']=='B']
    cm=lambda arm,c:S.mean(r[m] for r in R if r['arm']==arm and r['cycle']==c)
    print(m,'A mean %.3f sd %.3f | B mean %.3f sd %.3f | A-B %.3f (%.2f%%) | per-cycle A-B: %s'%(S.mean(a),S.pstdev(a),S.mean(b),S.pstdev(b),S.mean(a)-S.mean(b),100*(S.mean(a)-S.mean(b))/S.mean(b),['%.3f'%(cm('A',c)-cm('B',c)) for c in (0,1)]))
