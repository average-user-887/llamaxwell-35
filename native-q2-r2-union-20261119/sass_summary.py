"""Count static instructions in the largest backward branch of each 1024-byte specialization."""
import collections,json,re,sys
from pathlib import Path
out={}
for name,needle,path in zip(('control','union'),('ofa_sparse_attention_q2_strideILj1024','ofa_sparse_union_q2ILj1024'),sys.argv[1:],strict=True):
    bodies=Path(path).read_text().split('Function : ')
    body,=[s for s in bodies if s.startswith('_Z') and needle in s.splitlines()[0]]
    branches=[(int(a,16),int(b,16)) for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d\s+)?BRA (0x[0-9a-f]+)',body)]
    hi,lo=max(branches,key=lambda x:x[0]-x[1])
    ops=collections.Counter(z for a,z in re.findall(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9]*)',body) if lo<=int(a,16)<=hi)
    out[name]=dict(pc_start=hex(lo),pc_end=hex(hi),static_instructions=sum(ops.values()),opcodes=dict(sorted(ops.items())))
print(json.dumps(out,indent=2))
