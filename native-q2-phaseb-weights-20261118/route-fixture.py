"""CPU mirror of the two Phase-B work lists; enforce exactly one producer per lane-0 output."""
from collections import Counter
from random import Random

def validate(a,b):
    ranks=(a,b)
    shared=[(-1,-1)] + [(i,b.index(e)) for i,e in enumerate(a) if e in b]
    solo=[[i for i,e in enumerate(ranks[row]) if e not in ranks[1-row]] for row in (0,1)]
    outputs,loads = Counter(),Counter()
    for block in range(48):
        own=block%2
        for first in range(block*8,len(shared)*512,48*8):
            for warp in range(16):
                t=first+warp//2
                group,channel=divmod(t,512)
                rank0,rank1=shared[group]
                token=warp%2
                rank=(rank0,rank1)[token]
                exp=(-1 if rank<0 else ranks[token][rank])
                if group: assert a[rank0]==b[rank1]==exp
                outputs[token,rank,channel]+=1
                loads[exp,channel,'up' if token else 'gate']+=1
        for warp in range(16):
            for t in range(block//2*16+warp,len(solo[own])*512,24*16):
                idx,channel=divmod(t,512)
                rank=solo[own][idx]
                outputs[own,rank,channel]+=1
                for stage in ('gate','up'): loads[ranks[own][rank],channel,stage]+=1
    assert len(outputs)==2*9*512 and all(n==1 for n in outputs.values())
    assert all(outputs[row,rank,ch]==1 for row in (0,1) for rank in range(-1,8) for ch in range(512))
    assert len(loads)==(1+len(set(a)|set(b)))*512*2 and all(n==1 for n in loads.values())

rand=Random(17553)
for overlap in range(9):
    a=list(range(8))
    b=list(range(overlap))+list(range(16,24-overlap))
    rand.shuffle(a);rand.shuffle(b)
    validate(a,b);validate(b,a)
validate([128,215,73,136,110,167,197,112], [110,73,64,128,215,96,197,82])
print('CPU schedule verified at overlaps 0..8 with rank permutations and both token orders; every channel has exactly one producer and each distinct gate/up weight row one load.')
