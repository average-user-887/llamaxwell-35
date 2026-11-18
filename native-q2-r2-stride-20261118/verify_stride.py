"""Check address bounds and the logical split schedule across all admitted widths."""
import numpy as np

checked=0
for width in range(80128,83201,256):
    for pitch in (1024,2048):
        tokens=np.arange(0,width,4,dtype=np.uint32)
        j=np.arange(4,dtype=np.uint32)
        offsets=tokens[:,None]*np.uint32(pitch)+j[None,:]*np.uint32(pitch)
        old=(tokens.astype('uint64')[:,None]+j.astype('uint64')[None,:])*pitch
        assert np.array_equal(offsets,old)
        assert int(offsets.max())+512+31*16+15 < 2**32
        checked+=offsets.size
    # Include sparse partial tiles and full-attention invalid-plan fallback.
    for units in (*range(1,2113),width//4):
        for split in range(12):
            assert ((split+1)*units)<2**31
            assert int(np.int32(split*units)//12)==split*units//12
            assert int(np.int32((split+1)*units)//12)==(split+1)*units//12
print(f'Validated {checked} token byte offsets, both heads, both pitches and 12-way partition bounds.')
