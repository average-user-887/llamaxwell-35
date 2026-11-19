"""Check page coverage and tile accumulation against an independent CPU reference."""
import numpy as np

rng=np.random.default_rng(0x27197)

def pages(old, recent, end):
    return old+list(range(recent//256,(end+255)//256))

def descriptors(a,b):
    src=[a+[None]*(33-len(a)),b+[None]*(33-len(b))]
    return [(block,0 if block is None else
             (3 if block in b else 1) if r==0 else (0 if block in a else 2))
            for r,row in enumerate(src) for block in row]

cases=0
for width in range(80128,83201,256):
    for kind in range(8):
        for _ in range(4):
            e0=int(rng.integers(2,width+1))
            e1= [e0, min(width,e0+1),width,max(1,e0-20000),1,4097,4096,257][kind]
            ends=(e0,e1)
            rows=[]
            for end in ends:
                recent=max(0,end-4096)//256*256
                size=min(16,recent//256)
                old=sorted(rng.choice(recent//256,size=size,replace=False).tolist())
                rows.append(pages(old,recent,end))
            desc=descriptors(*rows)
            visits=np.zeros((2,width),np.uint8)
            for split in range(24):
                for t in range(split,(33+len(rows[1]))*16,24):
                    page,active=desc[t//16]
                    if not active: continue
                    lo=page*256+t%16*16
                    for row in range(2):
                        if active&(1<<row) and lo<ends[row]:
                            visits[row,lo:min(lo+16,ends[row])]+=1
            for row in range(2):
                expected=np.zeros(width,np.uint8)
                for page in rows[row]:
                    expected[256*page:min(ends[row],256*(page+1))]=1
                assert np.array_equal(visits[row],expected)
            cases+=1

# Exercise the same split/tile update on small random values including
# early/late/all-masked rows and extremely peaked logits. This tests the
# scheduled reduction's math; it is independent of CUDA code execution.
for scale in (0.1,1,90):
    v=rng.normal(size=(320,8)).astype('f')
    s=(rng.normal(size=(320,16))*scale).astype('f')
    s[:48,2]=-np.inf; s[:,7]=-np.inf; s[87:,8]=-np.inf
    sm=np.full((24,16),-np.finfo('f').max/2,dtype='f')
    den=np.zeros((24,16),dtype='f')
    out=np.zeros((24,16,8),dtype='f')
    for t in range(20):
        b=t%7  # multiple tiles per used split; more splits than nonempty tiles
        tile=s[t*16:t*16+16]
        m=np.maximum(sm[b],tile.max(axis=0)); a=np.exp2(sm[b]-m)
        w=np.exp2(tile-m)
        out[b]=a[:,None]*out[b]+w.T@v[t*16:t*16+16]
        den[b]=a*den[b]+w.sum(axis=0); sm[b]=m
    gain=np.exp2(sm-sm.max(axis=0)); d=(gain*den).sum(axis=0)
    result=(gain[:,:,None]*out).sum(axis=0)/np.where(d>0,d,1)[:,None]
    x=s.astype('d')
    m=x.max(axis=0); m[~np.isfinite(m)]=0
    w=np.exp2(x-m)
    ref=w.T@v/np.maximum(w.sum(axis=0),1.0)[:,None]
    assert np.all(np.isfinite(result)) and np.all(np.abs(result-ref)<=1e-5+1e-4*np.abs(ref))
print(f'{cases} two-row plan pairs across 13 widths had exact once-only coverage; masked and peaked tile reductions matched reference.')
