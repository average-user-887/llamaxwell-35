import hashlib,json,re,collections
from pathlib import Path
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(p):
    partial=re.split(r'Function : (\S+)\n',Path(p).read_text())
    return {name:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body) for name,body in zip(partial[1::2],partial[2::2])}
old,new=functions('control.sass'),functions('candidate.sass')
extra=set(new)-set(old)
assert len(old)==17 and len(extra)==1 and all(v and new[k]==v for k,v in old.items())
name,=extra
assert 'moe_q2_direct_down' in name
m=re.search('Function properties for '+name+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',Path('build.log').read_text())
assert m and m.groups()==('0','0','0','64','1','96')
pinned=digest('control-relinked.so')
assert pinned==digest('../resume-q2-moe-cohorts/libggml-cuda.so')=='19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00'
gdn=digest('../night-gdn-t2-q8-fast/occamy-gdn.cu.o')
assert gdn=='bcd063306ea65f5e8143a8d6501960dbfeda359f5b3a261d5d91d612c786eac0'
old_link,new_link=[l.split() for l in Path('compile.sh').read_text().splitlines() if l.startswith('/usr/bin/g++-14')]
assert [(a,b) for a,b in zip(old_link,new_link) if a!=b]==[('/work/control-relinked.so','/work/libggml-cuda.so'),('/q2/occamy-fused.cu.o','/work/occamy-fused.cu.o')]
assert old_link.count('/q8/occamy-gdn.cu.o')==1 and sum(x.endswith('.o') for x in old_link)==147
parts=re.split(r'Function : (\S+)\n',Path('candidate.sass').read_text());bodies=dict(zip(parts[1::2],parts[2::2]))
def counts(symbol,start,end):
    lines=[inst for pc,inst in re.findall(r'/\*([a-f0-9]+)\*/\s+([^;\n]+) ;',bodies[symbol]) if start<=int(pc,16)<=end]
    assert f'BRA 0x{start:x}' in lines[-1]
    return {'instructions':len(lines),**dict(collections.Counter(re.sub(r'^@!?P\d+\s+','',l).split()[0] for l in lines))}
c0=counts(name.replace('18moe_q2_direct_down','14moe_q2_cohorts'),0x48d8,0x6910)
c1=counts(name,0x4570,0x6218)
for opcode in ['FFMA.FTZ','FMUL.FTZ','PRMT','F2F.F32.F16','FADD.FTZ','SHFL.BFLY','LOP3.LUT']:
    assert c0[opcode]==c1[opcode]
assert c1.get('STS.128',0)==0 and c1['LDS.U.32']==2 and c0['LDS.U.32']==33
base=Path('occamy-fused.base.cu').read_text();src=Path('occamy-fused.cu').read_text()
def kernel(s,name):
    a=s.index('__global__ void __launch_bounds__(NT3, 2) '+name);b=s.index('\n}',a)+2
    return s[a:b]
a=kernel(base,'moe_q2_cohorts');b=kernel(src,'moe_q2_direct_down')
assert a[:a.index('    // C: down.')].replace('moe_q2_cohorts','moe_q2_direct_down')==b[:b.index('    // Exact Phase C:')]
# Reconstruct the original ldD->st72->lds9 mapping independently, including
# uint4 producer, expert crossing, per-lane word and both routed chunks.
checks=0
for shift in range(256):
 sel=[(shift+37*k)%256 for k in range(8)]
 for row in [0,1,23,24,383,384,2046,2047]:
  for chunk in range(2):
   for lane in range(32):
    for j in range(9):
     word=lane*9+j
     vector=word//4
     m=4*chunk+vector//18
     vector_offset=(vector%18)*16
     original=(sel[m]*2048+row)*288+vector_offset+(word%4)*4
     direct=(sel[4*chunk+(lane>>3)]*2048+row)*288+(lane&7)*36+j*4
     assert original==direct and direct%4==0
     checks+=1
for row in [0,1,23,24,383,384,2046,2047]:
 for lane in range(8):
  for j in range(9):
   word=lane*9+j
   original=row*288+(word//4)*16+(word%4)*4
   assert original==row*288+lane*36+j*4
   checks+=1
for physical_row in range(2):
 blocks=[p//2 for p in range(48) if p%2==physical_row]
 assert blocks==list(range(24))
 for tasks,down in [(4*257,False),(9*512,False),(2048,True)]:
  covering=[i for block in blocks for warp in range(16) for i in range(warp*24+block if down else block*16+warp,tasks,384)]
  assert sorted(covering)==list(range(tasks))
print(json.dumps({'native_q2_control_sha256':pinned,'candidate_cuda_sha256':digest('libggml-cuda.so'),'operator_sha256':digest('operator'),'q8_gdn_object_sha256':gdn,'objects_in_each_link':147,'device_functions_preserved':len(old),'device_instruction_words_preserved':sum(map(len,old.values())),'added_device_symbol':name,'candidate_static_device_words':len(new[name]),'phase_c_static_loop_opcodes':{'cohorts':c0,'directdown':c1},'registers':64,'static_shared_bytes':96,'dynamic_shared_bytes':39424,'stack_bytes':0,'spill_load_bytes':0,'spill_store_bytes':0,'phase_ab_source_identical':True,'aligned_lane_word_mappings_checked':checks,'threads_per_cta':512,'ctas_per_row':24,'rows_per_grid':2,'runtime_occupancy_required':48,'physical_executed':False},indent=2))
