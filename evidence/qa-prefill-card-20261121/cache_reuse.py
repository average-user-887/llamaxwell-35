import re,sys,json,collections
# per task: n_prompt (task.n_tokens), cache_n (first 'cached n_tokens' after new prompt), prompt eval ms/tokens
def parse(path):
    tasks=collections.OrderedDict(); cur=None; ev=collections.Counter()
    rx_new=re.compile(r'task (\d+) \| new prompt, .*task.n_tokens = (\d+)')
    rx_c=re.compile(r'task (\d+) \| cached n_tokens = (\d+)')
    rx_pe=re.compile(r'task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens')
    rx_sim=re.compile(r'checking sim = ([\d.]+) \((\d+)/(\d+)\)')
    sim=None
    for ln in open(path,errors='replace'):
        m=rx_sim.search(ln)
        if m: sim=(float(m[1]),int(m[2]),int(m[3]))
        for k in ('forcing full prompt re-processing','restored context checkpoint','erased invalidated context checkpoint','failed to truncate','n_past was set','cache_ram','prompt cache'):
            if k in ln: ev[k]+=1
        m=rx_new.search(ln)
        if m: t=int(m[1]); tasks[t]={'n_prompt':int(m[2]),'cache_n':None,'sim':sim,'full':False}; cur=t; sim=None; continue
        if 'forcing full prompt re-processing' in ln and cur in tasks: tasks[cur]['full']=True
        m=rx_c.search(ln)
        if m and int(m[1]) in tasks and tasks[int(m[1])]['cache_n'] is None: tasks[int(m[1])]['cache_n']=int(m[2])
        m=rx_pe.search(ln)
        if m and int(m[1]) in tasks: tasks[int(m[1])].update(pe_ms=float(m[2]),pe_tok=int(m[3]))
    return tasks,ev
for p in sys.argv[1:]:
    T,ev=parse(p); rows=[t for t in T.values() if 'pe_ms' in t]
    tot_ms=sum(r['pe_ms'] for r in rows); tot_tok=sum(r['pe_tok'] for r in rows); tot_prompt=sum(r['n_prompt'] for r in rows)
    # 'cold' = processed tokens > 50% of prompt ; reuse-miss = prompt shares prefix (sim>0.5) but cold
    cold=[r for r in rows if r['pe_tok']>0.5*r['n_prompt']]
    miss=[r for r in cold if r['sim'] and r['sim'][0]>0.5]
    cold_ms=sum(r['pe_ms'] for r in cold); miss_ms=sum(r['pe_ms'] for r in miss)
    print(json.dumps(dict(log=p.split('/')[-2],requests=len(rows),prompt_tok=tot_prompt,processed_tok=tot_tok,reused_tok=tot_prompt-tot_tok,
      prefill_s=round(tot_ms/1e3,1),cold_req=len(cold),cold_s=round(cold_ms/1e3,1),
      cold_with_sim_gt_0p5=len(miss),miss_s=round(miss_ms/1e3,1),
      warm_req=len(rows)-len(cold),warm_s=round((tot_ms-cold_ms)/1e3,1),events=dict(ev))))
    if '-v' in sys.argv[0:1]: pass
