import hashlib,json,re
from pathlib import Path
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(p):
    partial=re.split(r'Function : (\S+)\n',Path(p).read_text())
    return {name:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body) for name,body in zip(partial[1::2],partial[2::2])}
old,new=functions('control.sass'),functions('candidate.sass')
extra=set(new)-set(old)
assert len(old)==17 and len(extra)==1 and all(v and new[k]==v for k,v in old.items())
name,=extra
assert 'moe_q2_gate_up' in name
m=re.search('Function properties for '+name+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',Path('build.log').read_text())
assert m and m.groups()==('0','0','0','63','1','96')
pinned=digest('control-relinked.so')
assert pinned==digest('../resume-q2-moe-cohorts/libggml-cuda.so')=='19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00'
gdn=digest('../night-gdn-t2-q8-fast/occamy-gdn.cu.o')
assert gdn=='bcd063306ea65f5e8143a8d6501960dbfeda359f5b3a261d5d91d612c786eac0'
old_link,new_link=[l.split() for l in Path('compile.sh').read_text().splitlines() if l.startswith('/usr/bin/g++-14')]
diff=[(a,b) for a,b in zip(old_link,new_link) if a!=b]
assert diff==[('/work/control-relinked.so','/work/libggml-cuda.so'),('/q2/occamy-fused.cu.o','/work/occamy-fused.cu.o')]
assert old_link.count('/q8/occamy-gdn.cu.o')==1 and sum(x.endswith('.o') for x in old_link)==147
for row in range(2):
    blocks=[p//2 for p in range(48) if p%2==row]
    assert blocks==list(range(24))
    for tasks,down in [(4*257,False),(9*512,False),(2048,True)]:
        covering=[i for b in blocks for w in range(16) for i in range(w*24+b if down else b*16+w,tasks,384)]
        assert sorted(covering)==list(range(tasks))
def loop_counts(sass,symbol,start,end):
    parts=re.split(r'Function : (\S+)\n',Path(sass).read_text())
    body=dict(zip(parts[1::2],parts[2::2]))[symbol]
    lines=[inst for pc,inst in re.findall(r'/\*([a-f0-9]+)\*/\s+([^;\n]+) ;',body) if start <= int(pc,16) <= end]
    assert f'BRA 0x{start:x}' in lines[-1]
    return {opcode:sum(opcode in inst for inst in lines) for opcode in ('LDS.U.128','LDS.U.32','FFMA','PRMT','LDG','SHFL.BFLY')}
base_counts=loop_counts('candidate.sass',name.replace('gate_up','cohorts'),0x2038,0x37c8)
pair_counts=loop_counts('candidate.sass',name,0x2030,0x3708)
assert pair_counts['LDS.U.128']<base_counts['LDS.U.128'] and pair_counts['FFMA']==base_counts['FFMA']
print(json.dumps({'native_q2_control_sha256':pinned,'candidate_cuda_sha256':digest('libggml-cuda.so'),'operator_sha256':digest('operator'),
'q8_gdn_object_sha256':gdn,'objects_in_each_link':147,'device_functions_preserved':len(old),'device_instruction_words_preserved':sum(map(len,old.values())),
'added_device_symbol':name,'candidate_static_device_words':len(new[name]),'phase_b_static_loop_opcodes':{'cohorts':base_counts,'gateup':pair_counts},'registers':63,'static_shared_bytes':96,'dynamic_shared_bytes':39424,'stack_bytes':0,'spill_load_bytes':0,'spill_store_bytes':0,
'threads_per_cta':512,'ctas_per_row':24,'rows_per_grid':2,'runtime_occupancy_required':48},indent=2))
