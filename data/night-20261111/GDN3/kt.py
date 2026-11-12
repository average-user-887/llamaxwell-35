import csv,sys,collections,statistics as S
# usage: kt.py csv... : per-kernel (gdn_fused*, moe_v2*) count, median/mean us over last 150 calls, plus total GPU kernel us/token over last 5 tokens est.
for fn in sys.argv[1:]:
    rows=list(csv.reader(open(fn)))
    i=[k for k,r in enumerate(rows) if r and r[0]=='Start'][0]; h=rows[i]; u=rows[i+1]
    di=h.index('Duration'); ni=h.index('Name'); f={'ns':1e-3,'us':1,'ms':1e3,'s':1e6}[u[di]]
    ks=[(r[ni],float(r[di])*f) for r in rows[i+2:] if len(r)>ni and r[ni] and not r[ni].startswith('[CUDA')]
    g=collections.defaultdict(list)
    for n,d in ks:
        if 'gdn_fused' in n or 'moe_v2' in n: g[n.split('(')[0].replace('void ','')].append(d)
    for n,v in sorted(g.items()): print(fn.split('/')[-1], n, len(v), 'med %.1f mean %.1f' % (S.median(v[-150:]), S.mean(v[-150:])))
