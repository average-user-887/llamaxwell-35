import hashlib,json,re
from pathlib import Path
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(p):
    partial=re.split(r'Function : (\S+)\n',Path(p).read_text())
    return {name:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body) for name,body in zip(partial[1::2],partial[2::2])}
old,new=functions('control.sass'),functions('candidate.sass')
extra=set(new)-set(old)
assert len(old)==16 and len(extra)==1 and all(v and new[k]==v for k,v in old.items())
name,=extra
assert 'moe_q2_cohorts' in name
m=re.search('Function properties for '+name+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',Path('build.log').read_text())
assert m and m.groups()==('0','0','0','63','1','96')
pinned=digest('control-relinked.so')
assert pinned==digest('../night-gdn-t2-q8-fast/libggml-cuda.so')=='3b3c6db37769ee281b4bc357faf6016f34388d7270545d6b13bbc37777c79fdf'
gdn=digest('../night-gdn-t2-q8-fast/occamy-gdn.cu.o')
assert gdn=='bcd063306ea65f5e8143a8d6501960dbfeda359f5b3a261d5d91d612c786eac0'
old_link,new_link=[l.split() for l in Path('compile.sh').read_text().splitlines() if l.startswith('/usr/bin/g++-14')]
diff=[(a,b) for a,b in zip(old_link,new_link) if a!=b]
assert diff==[('/work/control-relinked.so','/work/libggml-cuda.so'),('CMakeFiles/ggml-cuda.dir/occamy-fused.cu.o','/work/occamy-fused.cu.o')]
assert old_link.count('/q8/occamy-gdn.cu.o')==1 and sum(x.endswith('.o') for x in old_link)==147
for row in range(2):
    blocks=[p//2 for p in range(48) if p%2==row]
    assert blocks==list(range(24))
    for tasks,down in [(4*257,False),(9*512,False),(2048,True)]:
        covering=[i for b in blocks for w in range(16) for i in range(w*24+b if down else b*16+w,tasks,384)]
        assert sorted(covering)==list(range(tasks))
print(json.dumps({'native_q8_control_sha256':pinned,'candidate_cuda_sha256':digest('libggml-cuda.so'),'operator_sha256':digest('operator'),
'q8_gdn_object_sha256':gdn,'objects_in_each_link':147,'device_functions_preserved':len(old),'device_instruction_words_preserved':sum(map(len,old.values())),
'added_device_symbol':name,'candidate_static_device_words':len(new[name]),'registers':63,'static_shared_bytes':96,'dynamic_shared_bytes':39424,'stack_bytes':0,'spill_load_bytes':0,'spill_store_bytes':0,
'threads_per_cta':512,'ctas_per_row':24,'rows_per_grid':2,'runtime_occupancy_required':48},indent=2))
