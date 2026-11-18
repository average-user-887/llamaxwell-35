"""Compare the warp's byte pairs and unsigned Q4 nibble mapping to independent scalar sums."""
import math
import random
import struct

rng=random.Random(20261118)
def half(x):
    return struct.unpack("<e",struct.pack("<e",x))[0]
def s16(x):
    return (x&65535)-65536 if x&32768 else x&65535
def mad2(x,y):
    return s16(x)*s16(y)+s16(x>>16)*s16(y>>16)
def prmt(a,b):
    return int.from_bytes((a.to_bytes(4,'little')+b.to_bytes(4,'little'))[2:6],'little')
for case in range(7):
    x=[rng.uniform(-2,2) for _ in range(2048)]
    if case==0:x=[0.]*2048
    if case==1:x=([1.,-1.]*1024)
    if case==2:x=[0. if i%32!=31 else -1. for i in range(2048)]
    if case==3:x=[rng.randrange(-127,128)/127. for _ in range(2048)]
    qq=[]; sums=[]; inputs=[]; packed=[[None]*4 for _ in range(8*32)]
    for b in range(64):
        row=x[32*b:32*b+32]
        d=max(map(abs,row))/127.
        quant=[math.copysign(math.floor(abs(v/d)+0.5),v) if d else 0 for v in row]
        quant=[int(v) for v in quant]
        qq.append(quant);sums.append((half(d),half(sum(row))))
        for lane in range(32):
            if not (lane&2):
                word=((b&1)*4+((lane&15)>>2))*32+(b>>1)
                field=((lane>>4)*2)+(lane&1)
                assert packed[word][field] is None
                packed[word][field]=(quant[lane]&65535)|((quant[lane+2]&65535)<<16)
        nib=[rng.randrange(16) for _ in range(32)]
        bits=bytes(a+(b<<4) for a,b in zip(nib[:16],nib[16:]))
        inputs.append((bytes(struct.pack('<e',rng.uniform(-.5,.5)))+bits,nib))
    assert all(None not in line for line in packed)
    for lane in range(32):
        data=b''.join(inputs[2*lane+b][0] for b in range(2));w=struct.unpack('<9I',data)
        for b in range(2):
            even=odd=0
            for j in range(4):
                v=prmt(w[j],w[j+1]) if b==0 else w[j+5]
                a=packed[(b*4+j)*32+lane]
                even+=mad2(v&0x000f000f,a[0])+mad2((v>>4)&0x000f000f,a[2])
                odd+=mad2((v>>8)&0x000f000f,a[1])+mad2((v>>12)&0x000f000f,a[3])
            idx=2*lane+b
            assert even+odd==sum(q*n for q,n in zip(qq[idx],inputs[idx][1]))
            bits=(w[0] if b==0 else w[4]>>16)&65535
            assert bits.to_bytes(2,'little')==inputs[idx][0][:2]
print('CPU layout passed 448 complete Q4 blocks, including zero, alternating signed extremes and sparse end lanes.')
