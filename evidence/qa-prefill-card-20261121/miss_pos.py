import re,sys
rx_new=re.compile(r'task (\d+) \| new prompt, .*task.n_tokens = (\d+)')
rx_chk=re.compile(r'task (\d+) \| checking checkpoint with \[(\d+), \d+\] against (\d+)')
rx_sim=re.compile(r'checking sim = ([\d.]+) \((\d+)/(\d+)\)')
for p in sys.argv[1:]:
    hist=[]; cur=None; miss=[]; sim=None; chks=[]
    for ln in open(p,errors='replace'):
        m=rx_sim.search(ln)
        if m: sim=(int(m[2]),int(m[3]))
        m=rx_new.search(ln)
        if m: cur=int(m[2]); hist.append(cur); chks=[]; continue
        m=rx_chk.search(ln)
        if m: chks.append(int(m[2])); div=int(m[3])
        if 'forcing full prompt' in ln:
            # earlier prompt ends in same conversation (monotone run) that are <= divergence
            prior=[h for h in hist[-12:-1] if h<=div]
            miss.append((div,cur,chks[:],max(prior) if prior else None))
    gap=[d-pr for d,_,_,pr in miss if pr]
    print(p.split('/')[-2],'misses',len(miss),'with an earlier same-run prompt end <= divergence:',len(gap),'median gap tok',sorted(gap)[len(gap)//2] if gap else None)
    for x in miss[:6]: print('  div',x[0],'new',x[1],'ckpts',x[2],'best earlier end',x[3])
    lost=[x[1]-x[0] for x in miss]; print('  tokens after divergence (median)',sorted(lost)[len(lost)//2],'div/new median',sorted(round(x[0]/x[1],2) for x in miss)[len(miss)//2])
