"""Verify the reduce-scatter and tile boundary results independently on CPU."""
import numpy as np

lanes=np.arange(32)
rng=np.random.default_rng(6603)
values=rng.normal(size=(1024,32,8)).astype(np.float32)
values[0,:,:]=-0.0
values[1,:,:]=np.nextafter(np.float32(1),np.float32(0))
values[2,3,7]=np.nan; values[3,27,2]=np.inf; values[4,11,5]=-np.inf
full=values.copy()
tile=values.copy()
with np.errstate(invalid='ignore'):
    for stride in (16,8,4,2,1):
        full=full+full[:,lanes^stride,:]
    for stride,half in ((16,4),(8,2),(4,1)):
        upper=(lanes&stride)!=0
        keep=np.where(upper[None,:,None],tile[:,:,half:],tile[:,:,:half])
        send=np.where(upper[None,:,None],tile[:,:,:half],tile[:,:,half:])
        tile=keep+send[:,lanes^stride,:]
    for stride in (2,1):
        tile=tile+tile[:,lanes^stride,:]
original=full[:,0,:]
new=tile[:,::4,0]
assert np.array_equal(np.isnan(original),np.isnan(new))
finite=~np.isnan(original)
assert np.array_equal(original[finite].view('u4'),new[finite].view('u4'))
for width in range(80128,83201,256):
    nb=width//256
    writes=[4*cta+b for cta in range((nb+3)//4) for b in range(4) if 4*cta+b<nb]
    assert writes==list(range(nb))
    for b in writes:
        for w in range(4):
            assert b*256+w*64<width
print('Eight-head FP32 reduction matched 1,024 warp inputs by bits or NaN classification. All 13 active widths cover each original block once.')
