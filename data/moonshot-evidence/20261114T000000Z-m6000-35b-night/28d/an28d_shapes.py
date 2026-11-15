import sys,re,collections
src=open(sys.argv[1]).read().split("K = load(sys.argv[1])")[0]
sys.argv=[sys.argv[0],sys.argv[2]]
exec(src)
K=load(sys.argv[1]); GD='gated_delta_net_cuda'; segs=collections.defaultdict(list); g=0
for k in K:
    segs[g//30].append(k)
    if k[2].startswith(GD): g+=1
for i in (15,45):
    c=collections.defaultdict(lambda:[0,0.0])
    for k in segs[i]:
        if k[2].startswith('DENSE') or k[2].startswith('dequant') or k[2].startswith('mul_mat_q') or 'sgemm' in k[2]:
            key=(k[2][:60],k[3]); c[key][0]+=1; c[key][1]+=k[1]
    print('seg',i)
    for key,v in sorted(c.items(),key=lambda x:-x[1][1])[:16]: print(v[0],'%.1f ms'%v[1],key)
