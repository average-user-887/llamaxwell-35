"""Exhaust output scheduling, prove ordered sum trees, and compare seeded FP32 bit patterns."""
import ctypes
import json
import random
import struct

H, CTAS, WARPS = 2048, 24, 16
rows = [r + offset for cta in range(CTAS) for warp in range(WARPS)
        for r in range(4*(warp*CTAS+cta),H,4*CTAS*WARPS) for offset in range(4)]
assert sorted(rows) == list(range(H))
# These are byte locations of the original nine 4-byte weights for a 64-element unit.
for quartet in range(H//4):
    for lane in range(32):
        row = 4*quartet + lane//8
        unit = lane % 8
        for word in range(9):
            assert 4*quartet*288 + 4*(9*lane+word) == row*288 + 36*unit + 4*word
for k in range(8):
    for lane in range(32):
        assert k*8 + lane%8 == k*8 + (lane & 7)
for k in range(256):
    assert (k*H*288) % 16 == 0

def butterfly(values, steps, add):
    values = list(values)
    for s in steps:
        values = [add(v,values[i^s]) for i,v in enumerate(values)]
    return values[0]

def old(values, add):
    return butterfly(values,[16,8,4,2,1],add)

def new(values, add):
    partial = [add(add(values[i],values[i+16]),add(values[i+8],values[i+24])) for i in range(8)]
    return butterfly(partial,[4,2,1],add)

assert old(range(32), lambda a,b:(a,b)) == new(range(32), lambda a,b:(a,b))
f32 = lambda x: ctypes.c_float(x).value
fma = ctypes.CDLL('libm.so.6').fmaf
fma.argtypes = [ctypes.c_float]*3
fma.restype = ctypes.c_float
bits = lambda x: struct.pack('<f',x)
rng = random.Random(26784)
def random_value():
    return f32(rng.uniform(-1,1) * 2.**rng.randrange(-20,20))
for _ in range(256):
    da, db, ds = [[random_value() for _ in range(32)] for k in range(3)]
    weights = [f32(rng.random()) for k in range(9)]
    old_input = [fma(weights[lane//8],da[lane],f32(weights[4+lane//8]*db[lane])) for lane in range(32)]
    for lane in range(8):
        old_input[lane] = fma(weights[8],ds[lane],old_input[lane])
    def new_value(k, group):
        i = 8*k + group
        value = fma(weights[k],da[i],f32(weights[k+4]*db[i]))
        return fma(weights[8],ds[group],value) if k == 0 else value
    new_input = [new_value(i//8,i&7) for i in range(32)]
    assert bits(old(old_input,lambda a,b:f32(a+b))) == bits(new(new_input,lambda a,b:f32(a+b)))
print(json.dumps(dict(checked_rows=len(rows), down_words_per_expert=2048*8*9,
    exact_original_lane0_sum_tree=True, seeded_weighted_f32_reductions=256,
    original_warp_dots_per_token=2048*3, candidate_warp_dots_per_token=512*9,
    original_down_weight_bytes_per_token=2048*288*9,
    candidate_down_weight_bytes_per_token=512*1152*9,
    candidate_full_warp_dots_per_output_quartet=9, control_warp_dots_per_output_quartet=12)))
