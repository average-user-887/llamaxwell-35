import hashlib,json,re
from pathlib import Path
c=Path(__file__).resolve().parent
p=c.parent/'finish33-r2-stride'
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def funcs(path):
    a=re.split(r'Function : (\S+)\n',path.read_text())
    return {name:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body) for name,body in zip(a[1::2],a[2::2])}
base='369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871'
assert sha(c/'control-relinked.so')==sha(p/'libggml-cuda.so')==base
old,new=funcs(c/'control.sass'),funcs(c/'candidate.sass')
assert old==funcs(p/'candidate.sass')
assert len(old)==37 and all(v and v==new[k] for k,v in old.items())
extra=sorted(new.keys()-old.keys())
assert len(extra)==5 and all('union' in name for name in extra)
resources={}
for name in extra:
    log=(c/'build.log').read_text()
    m=re.search(rf'Function properties for {name}\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',log)
    assert m,name
    stack,st,ld,regs,bars,smem=map(int,m.groups())
    assert stack==st==ld==0 and regs<=128 and smem<=49152,name
    resources[name]=dict(regs=regs,stack=stack,spill_store=st,spill_load=ld,shared_bytes=smem,static_barrier_slots=bars)
commands=(c/'compile.sh').read_text().splitlines()
assert commands[2].replace('control-relinked.so','libggml-cuda.so').replace('/fast/fattn-occamy.cu.o','/work/fattn-occamy.cu.o')==commands[4]
print(json.dumps(dict(control_sha256=base,control_object=sha(p/'fattn-occamy.cu.o'),candidate_sha256=sha(c/'libggml-cuda.so'),candidate_object=sha(c/'fattn-occamy.cu.o'),harness_sha256=sha(c/'test-backend-ops'),preserved_functions=len(old),preserved_words=sum(len(v) for v in old.values()),resources=resources,scope='CPU build and device code audit; physical outputs and timing pending'),indent=2))
