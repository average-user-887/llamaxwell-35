import sys, os, importlib.util, numpy as np
N='<LAB>/occamy35-night-20261113'
sp=importlib.util.spec_from_file_location('an',N+'/NC2/g1c_analyze.py'); an=importlib.util.module_from_spec(sp); sp.loader.exec_module(an)
RUN=N+'/N2/run'
roots={'B':RUN,'Ap':RUN,'Q8KV':N+'/Q8S/long','Q8K':N+'/Q8S/long'}
Bk=an.BUCKETS
def doc(arm,k,B):
    r=roots[arm]; a=an.load(RUN,'A',k,B); x=an.load(r,arm,k,B)
    if x is None: return None
    kl,st=an.kl_rows(a,x); return kl.mean(),100*(st.mean()-1),len(kl)
out=[]
for arm in ('Ap','B','Q8KV','Q8K'):
    out.append('\n## %s vs A (per doc: dmean / top pt)\n'%arm); out.append('| doc | 32K | 65K | 98K |\n|---|---|---|---|')
    per={B:[] for B in Bk}
    for k in range(8):
        cells=[]
        for B in Bk:
            r=doc(arm,k,B)
            if r is None: cells.append('-'); continue
            per[B].append(r); cells.append('%+.5f / %+.2f'%(r[0],r[1]))
        out.append('| %d | %s |'%(k,' | '.join(cells)))
    out.append('\n| pooled | docs | rows | dmean | CI95 doc | top pt | CI95 doc top |\n|---|---|---|---|---|---|---|')
    for B in Bk:
        v=per[B]
        if not v: continue
        d=[x[0] for x in v]; t=[x[1] for x in v]; c=an.ci(d); ct=an.ci(t)
        out.append('| %d | %d | %d | %+.5f | [%+.5f,%+.5f] | %+.2f | [%+.2f,%+.2f] |'%(B,len(v),sum(x[2] for x in v),np.mean(d),c[0],c[1],np.mean(t),ct[0],ct[1]))
print('\n'.join(out))
