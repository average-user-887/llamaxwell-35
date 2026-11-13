import csv, sys, json, re, collections
def load(d):
    rows=[]; hdr=None; units=None
    for r in csv.reader(open(d+'/trace.csv')):
        if r and r[0]=='Start': hdr=r; continue
        if hdr and units is None: units=r; continue
        if hdr and len(r)==len(hdr): rows.append(r)
    i=hdr.index
    sc={'s':1,'ms':1e-3,'us':1e-6}[units[0]]; dc={'s':1,'ms':1e-3,'us':1e-6}[units[1]]
    return [(float(r[i('Start')])*sc,float(r[i('Duration')])*dc,r[i('Name')],float(r[i('Size')] or 0) if r[i('Size')] else 0) for r in rows]
def group(n):
    l=n.lower()
    if 'memcpy' in l or 'memset' in l: return 'quantize/copy'
    if 'flash_attn' in l or 'fattn' in l or 'fa_' in l or 'occamy_fa' in l or 'maxwell' in l and 'attn' in l: return 'FA/attention'
    if 'gdn' in l or 'ssm' in l or 'gated_delta' in l or 'conv' in l: return 'GDN/ssm'
    if 'moe' in l or 'mul_mat_id' in l or 'mmid' in l or 'mmq_id' in l or 'topk' in l or 'argsort' in l: return 'MoE matmul'
    if 'mul_mat_q' in l or 'mmq' in l or 'mul_mat_vec' in l or 'mmvq' in l or 'gemm' in l or 'sgemm' in l or 'cublas' in l or 'rowpack' in l: return 'dense mmq/mmvq'
    if 'quantize' in l or 'cpy' in l or 'copy' in l or 'dup' in l or 'concat' in l or 'get_rows' in l or 'set_rows' in l or 'q8' in l: return 'quantize/copy'
    return 'other'
def clusters(ev, gap=2.0):
    ev=sorted(ev); cl=[[ev[0]]]
    for e in ev[1:]:
        if e[0]-(cl[-1][-1][0]+cl[-1][-1][1])>gap: cl.append([])
        cl[-1].append(e)
    return cl
if __name__=='__main__':
    d=sys.argv[1]; ev=load(d); cl=clusters(ev)
    for k,c in enumerate(cl): print(k,len(c),round(c[0][0],2),round(c[-1][0]+c[-1][1]-c[0][0],3),round(sum(e[1] for e in c),3))
