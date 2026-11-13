import json, collections, sys, re
import n1_an as a
def group(n):
    l=n.lower()
    if 'memcpy' in l or 'memset' in l: return 'quantize/copy'
    if 'flash_attn' in l or 'fattn' in l or 'maxwell_fa' in l or 'gemm_attn' in l: return 'FA/attention'
    if 'gated_delta' in l or 'ssm_conv' in l or 'gdn' in l: return 'GDN/ssm'
    if re.search(r'mul_mat_q<|mm_ids_helper|mmq_make_tail|moe|topk_moe|occamy::|_id',l): return 'MoE matmul'
    if re.search(r'sgemm|gemv|gemm|cublas|mul_mat_vec|head_rowpack',l): return 'dense mmq/mmvq'
    if re.search(r'quantize|dequant|cpy|copy|dup|concat|get_rows|set_rows|hg_gather',l): return 'quantize/copy'
    return 'other'
GR=['FA/attention','MoE matmul','dense mmq/mmvq','GDN/ssm','quantize/copy','other']
def short(n):
    n=re.sub(r'^void ','',n); n=re.sub(r'_INTERNAL_[0-9a-f_]+','',n); n=re.sub(r'\((char|float|void|__half|const|int|unsigned|ggml)[^)]*\)?.*$','',n); return n[:90]
def agg(c):
    t=collections.defaultdict(lambda:[0.0,0])
    for e in c: t[e[2]][0]+=e[1]; t[e[2]][1]+=1
    return t
def phases(arm):
    d='run/arm'+arm; cl=a.clusters(a.load(d)); st=json.load(open(d+'/steps.json'))
    idx={}; n=len(st)
    for j in range(1,n): idx[(st[j]['cell'],st[j]['phase'])]=len(cl)-(n-j)
    res={}
    for s in st:
        key=(s['cell'],s['phase'])
        if key in idx:
            c=cl[idx[key]]; span=c[-1][0]+c[-1][1]-c[0][0]
            res[key]=dict(c=c,span=span,wall=s['wall'],step=s)
    return res
def tab(res,cell,ph):
    r=res[(cell,ph)]; t=agg(r['c']); tot=sum(v[0] for v in t.values())
    return r,t,tot
def main():
    arms=sys.argv[1:]; L=[]; data={}
    for arm in arms:
        res=phases(arm); data[arm]=res
    # summary table
    L.append('| arm | append | rep | wall s (HTTP) | prefill s (server) | GPU span s | GPU busy s | kernels |\n|---|---|---|---|---|---|---|---|')
    for arm in arms:
        for rep in (1,2):
            for sz in ('1K','4K'):
                cell='r%d_%s'%(rep,sz); r,t,tot=tab(data[arm],cell,'prefill')
                L.append('| %s | +%s | %d | %.2f | %.2f | %.2f | %.2f | %d |'%(arm,sz,rep,r['wall'],r['step']['prompt_ms']/1e3,r['span'],tot,len(r['c'])))
    # grouped shares
    L.append('\n### Grouped shares of GPU busy time in prefill (% ; ms)\n')
    L.append('| arm | append | rep | '+' | '.join(GR)+' |\n|---|---|---|'+'---|'*len(GR))
    for arm in arms:
        for sz in ('1K','4K'):
            for rep in (1,2):
                r,t,tot=tab(data[arm],'r%d_%s'%(rep,sz),'prefill'); g=collections.defaultdict(float)
                for n,v in t.items(): g[group(n)]+=v[0]
                L.append('| %s | +%s | %d | '%(arm,sz,rep)+' | '.join('%.1f%% (%.0f)'%(100*g[x]/tot,g[x]*1e3) for x in GR)+' |')
    # top10 per arm x size (rep 1 and rep 2 totals averaged? show rep1, report rep2 total ms alongside)
    for arm in arms:
        for sz in ('1K','4K'):
            L.append('\n### Top 10 kernels, arm %s, +%s (rep1 | rep2 ms)\n'%(arm,sz))
            r1,t1,tot1=tab(data[arm],'r1_'+sz,'prefill'); r2,t2,tot2=tab(data[arm],'r2_'+sz,'prefill')
            L.append('| # | kernel | group | rep1 ms | rep2 ms | % of prefill GPU busy (rep1) | % of prefill wall (rep1) | calls |\n|---|---|---|---|---|---|---|---|')
            for i,(n,v) in enumerate(sorted(t1.items(),key=lambda x:-x[1][0])[:10]):
                L.append('| %d | `%s` | %s | %.1f | %.1f | %.1f | %.1f | %d |'%(i+1,short(n),group(n),v[0]*1e3,t2.get(n,[0,0])[0]*1e3,100*v[0]/tot1,100*v[0]/r1['step']['wall'],v[1]))
            L.append('(prefill GPU idle gaps rep1: %.2f s of %.2f s span)'%(r1['span']-tot1,r1['span']))
    # restore & probe
    L.append('\n### Restore and first-decode-step (1-token batch) phases\n')
    L.append('| arm | cell | restore HTTP s | restore GPU busy s | restore kernels/copies | 1-token step HTTP s | 1-token step GPU busy ms | top kernel of 1-token step |\n|---|---|---|---|---|---|---|---|')
    for arm in arms:
        for cell in ('r1_1K','r1_4K','r2_1K','r2_4K'):
            rr,rt,rtot=tab(data[arm],cell,'restore'); pr,pt,ptot=tab(data[arm],cell,'probe')
            top=sorted(pt.items(),key=lambda x:-x[1][0])[0]
            L.append('| %s | %s | %.2f | %.3f | %d | %.3f | %.1f | `%s` %.1f ms |'%(arm,cell,rr['wall'],rtot,len(rr['c']),pr['wall'],ptot*1e3,short(top[0]),top[1][0]*1e3))
    open('REPORT_TABLES.md','w').write('\n'.join(L)+'\n'); print('\n'.join(L))
main()
