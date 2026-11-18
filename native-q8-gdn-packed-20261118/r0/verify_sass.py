import hashlib
import json
import re
from collections import Counter
from pathlib import Path

def functions(path):
    seg=re.split(r'Function : (\S+)\n',Path(path).read_text())
    return dict(zip(seg[1::2],seg[2::2]))

def words(body):
    return re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body)

def mainloop(body):
    instructions=[(int(a,16),op,arg) for a,op,arg in re.findall(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+) (.*?);',body)]
    for end,opcode,arg in instructions:
        if opcode!='BRA' or not arg.strip().startswith('0x'): continue
        start=int(arg.strip(),16)
        ops=Counter(op for pc,op,_ in instructions if start<=pc<=end)
        if ops['XMAD.S16.S16']!=128: continue
        return {'start':hex(start),'end':hex(end),'counts':dict(ops)}
    raise ValueError('missing both-row XMAD loop')

old,new=functions('old.sass'),functions('new.sass')
parent='_ZN10occamy_gdn10gdn_fused2ILi2EEEvNS_7Params2E'
packed=parent.replace('Li2EEE','Li3EEE')
assert all(words(v) and words(v)==words(new[k]) for k,v in old.items())
added=set(new)-set(old)
assert len(added)==2 and packed in added and any('q8pa_pack_rows' in k for k in added)
a,b=mainloop(old[parent]),mainloop(new[packed])
for op in ['XMAD.S16.S16','FFMA.FTZ','SHFL.BFLY','PRMT']:
    assert a['counts'].get(op)==b['counts'].get(op),op
assert b['counts'].get('LDS.U.32',0)<a['counts'].get('LDS.U.32',0)
assert sum(v for k,v in b['counts'].items() if k.startswith('STS'))<sum(v for k,v in a['counts'].items() if k.startswith('STS'))
log=Path('build.log').read_text()
r=re.search('Function properties for '+re.escape(packed)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',log)
res=list(map(int,r.groups()))
assert res[:3]==[0,0,0] and res[3]<=80 and res[4:]==[1,2512],res
print(json.dumps({'preserved_device_functions':len(old),'added_device_functions':sorted(added),'regs':res[3],'stack_bytes':0,'spill_store_bytes':0,'spill_load_bytes':0,'dynamic_shared_bytes':27392,'static_shared_bytes':2512,'phase_A_loop_old':a,'phase_A_loop_new':b},indent=2))
