import hashlib,struct,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent
so=h.parent/'resume-q2-moe-cohorts/libggml-cuda.so'
b=so.read_bytes();assert hashlib.sha256(b).hexdigest()=='19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00'
assert b[:6]==b'\x7fELF\x02\x01'
ph,=struct.unpack_from('<Q',b,32); ent,n=struct.unpack_from('<HH',b,54)
rw=[struct.unpack_from('<IIQQQQQQ',b,ph+i*ent) for i in range(n)]
assert [(p[2],p[3],p[5],p[6]) for p in rw if p[:2]==(1,6)]==[(0x2236f00,0x2237f00,0x6858,0x189c8)]
sym=subprocess.check_output(['nm','-an',so],text=True)
for off,name in [(0x2244110,'q2_rows'),(0x2244120,'q2_calls'),(0x2244100,'q2_skips')]:assert any(line.startswith(f'{off:016x} b ') and name in line for line in sym.splitlines()),name
subprocess.run([h/'supervise','--cpu-check',so],check=True)
print('Pinned ELF writable PT_LOAD geometry and four fixed owned-child atomic addresses verified.')
