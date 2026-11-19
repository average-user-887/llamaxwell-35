"""Build the inherited, bounded owned-child supervisor for the exact down4 ELF."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess

here = Path(__file__).resolve().parent
library = here.parent / 'break100-q2-down4/libggml-cuda.so'
original = (here / 'supervise.original.c').read_text()
observed = library.read_bytes()
assert hashlib.sha256(observed).hexdigest() == 'b71f011fbe1448687ec20c465ebc235f4190b401dc1f3603cf8f0211cfa5673f'
ph, = struct.unpack_from('<Q',observed,32)
size,count = struct.unpack_from('<HH',observed,54)
headers = [struct.unpack_from('<IIQQQQQQ',observed,ph+i*size) for i in range(count)]
rw, = [p for p in headers if p[:2]==(1,6)]
assert (rw[2],rw[3],rw[5],rw[6],rw[7]) == (0x2243ec0,0x2244ec0,0x68d8,0x18b08,0x1000)
names = subprocess.check_output(['nm','-aSn',library],text=True).splitlines()
symbols = [(0x2251210,16,'_ZN6occamyL7q2_rowsE'),(0x2251220,8,'_ZN6occamyL8q2_callsE'),
    (0x2251208,8,'_ZN6occamyL8q2_skipsE'),(0x224cc10,24,'_ZL16ofa_sparse_calls'),
    (0x224cb2c,4,'_ZZ17ofa_sparse_launchILi2EEvR25ggml_backend_cuda_contextP11ggml_tensorE11stride_seen'),
    (0x2251200,8,'_ZN6occamyL11down4_callsE')]
for addr,sz,symbol in symbols:
    assert f'{addr:016x} {sz:016x} b {symbol}' in names
changed = original
for a,b in [('0x224d190,0x224d198,0x224d1a0,0x224d180,0x2248c00,0x2248b0c','0x2251210,0x2251218,0x2251220,0x2251208,0x224cc20,0x224cb2c,0x2251200'),
            ('0x2240ee0, vaddr 0x2240ee0, filesz 0x6898','0x2243ec0, vaddr 0x2244ec0, filesz 0x68d8'),
            ('0x2240ee0+0x18a68','0x2244ec0+0x18b08'),('memsz 0x18a68','memsz 0x18b08'),
            ('base+0x2248000','base+0x224c000'),('base+0x2240000','base+0x2244000'),
            ('fd,0x2240000','fd,0x2243000'),('0x225a000','0x225e000'),
            ('index>=6','index>=7'),('uint64_t v[6]','uint64_t v[7]'),('i<6','i<7'),('word(6,real','word(7,real'),
            ('memcpy(base+offsets[5],count_and_guard,sizeof count_and_guard);','memcpy(base+offsets[5],count_and_guard,sizeof count_and_guard);\n        *(uint64_t*)(base+offsets[6])=107;'),
            ('six exact words','seven exact words'),('PRIu64"}','PRIu64",\\"down4\\":%"PRIu64"}'),('v[4],v[5]);','v[4],v[5],v[6]);')]:
    assert a in changed,a
    changed=changed.replace(a,b)
import sys
if '--check' in sys.argv:
    assert (here/'supervise.c').read_text()==changed
else:
    (here/'supervise.c').write_text(changed)
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-o','supervise','supervise.c'],cwd=here,check=True)
subprocess.run([here/'supervise','--cpu-check',library],check=True)
print(json.dumps({'candidate_sha256':hashlib.sha256(observed).hexdigest(),'rw_program_header':rw,
                  'verified_symbols':symbols},indent=2))
