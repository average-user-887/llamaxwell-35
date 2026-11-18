"""Verify the pinned link, original device instructions, new loop structure and address bounds."""
import collections
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, description):
    if not ok:
        raise ValueError(description)


def functions(path):
    parts = re.split(r'Function : (\S+)\n', Path(path).read_text())
    return dict(zip(parts[1::2], parts[2::2]))


def words(body):
    return re.findall(r'/\* (0x[0-9a-f]{16}) \*/', body)


def loop_counts(body):
    instructions = [(int(pc, 16), line) for pc, line in re.findall(r'/\*([0-9a-f]+)\*/\s+([^;\n]+) ;', body)]
    loops = []
    for pc, line in instructions:
        m = re.search(r'\bBRA 0x([0-9a-f]+)', line)
        if m and pc - int(m[1], 16) > 0x1000:
            start = int(m[1], 16)
            ops = [re.sub(r'^@!?P\d+\s+', '', inst).split()[0] for addr, inst in instructions if start <= addr <= pc]
            counts = collections.Counter(op.split('.')[0] for op in ops)
            loops.append({'start_pc':hex(start),'end_pc':hex(pc),'instructions':len(ops),
                          'opcodes': dict(sorted(counts.items())), 'IADD.X':ops.count('IADD.X')})
    require(len(loops) == 2, "expected two large backward loops")
    return dict(zip(('B','C'), loops))


expected_cuda = '369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871'
control = sha('control-relinked.so')
require(control == expected_cuda == sha('../finish33-r2-stride/libggml-cuda.so'), 'current CUDA link identity')
require(sha('occamy-fused.base.cu') == sha('../resume-q2-moe-cohorts/occamy-fused.cu') == '348038adec471d26d19c2ec74eddbfeb0b2a06f670ba9f5e56393b4fef4d55fd', 'original source')
require(sha('../resume-q2-moe-cohorts/occamy-fused.cu.o') == 'bbea6f5c611282090050592a6e49f2c15238c104eb1b1420cb23e562220176eb', 'original object')
attention = sha('../finish33-r2-stride/fattn-occamy.cu.o')
gdn = sha('../night-gdn-t2-q8-fast/occamy-gdn.cu.o')
server = sha('../finish33-sampler-r1/libllama-server-impl.so')
require(attention == 'c2a805af82cf2060c84a105c44167e66631dfa30ab1cc8807c1b010188b0160a', 'fixed-stride attention object')
require(gdn == 'bcd063306ea65f5e8143a8d6501960dbfeda359f5b3a261d5d91d612c786eac0', 'Q8 GDN object')
require(server == '4a0ff82df62550ccb27927bd98a39d036148b4666abc819bec18f14686efcc64', 'current server')
control_link, candidate_link = [line.split() for line in Path('compile.sh').read_text().splitlines() if line.startswith('/usr/bin/g++-14')]
require(len(control_link)==len(candidate_link) and sum(x.endswith('.o') for x in control_link)==147, 'link input count')
require([(a,b) for a,b in zip(control_link,candidate_link) if a!=b] == [('/work/control-relinked.so','/work/libggml-cuda.so'),('/q2/occamy-fused.cu.o','/work/occamy-fused.cu.o')], 'link change')
require(all(control_link.count(p)==1 for p in ('/stride/fattn-occamy.cu.o','/q8/occamy-gdn.cu.o')), 'common object inputs')
old = functions('control.sass')
new = functions('candidate.sass')
old_sym = '_ZN6occamy14moe_q2_cohortsENS_6ParamsEb'
sym = '_ZN6occamy14moe_q2_offsetsENS_6ParamsEb'
require(len(old)==17 and set(new)==(set(old)|{sym}), 'device function inventory')
require(all(words(body) and words(body)==words(new[name]) for name,body in old.items()), 'preserved SASS instruction and control words')
require(re.search(r'Function properties for '+sym+r'\n\s*0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info\s*: Used 63 registers, used 1 barriers, 128 bytes smem',Path('build.log').read_text()), 'candidate build residency')
base_counts, cand_counts = loop_counts(new[old_sym]), loop_counts(new[sym])
for phase in base_counts:
    x, y = base_counts[phase], cand_counts[phase]
    for op in ('FFMA','FADD','FMUL','F2F','PRMT','SHFL','MUFU','RRO','LDG'):
        require(x['opcodes'].get(op)==y['opcodes'].get(op), phase+' floating-point, shuffles, unpack or loads: '+op)
    require(y['instructions'] < x['instructions'] and y['opcodes']['XMAD'] < x['opcodes']['XMAD'], phase+' narrow addressing')
# A complete factorization covers every expert, row and vector without allocating tensors.
pitch = 589824
for rows, stride in [(512,1152),(2048,288)]:
    vectors = [off for off in range(0,stride,16)]
    if rows==512:
        mapped = [(lane+32*i)*16 for lane in range(32) for i in range(3) if i<2 or lane<8]
    else:
        mapped = [(lane+32*i)%18*16 for lane in range(32) for i in range(3) if i<2 or lane<8]
    require(sorted(set(mapped))==vectors, 'vector offsets')
    for expert in range(256):
        for row in range(rows):
            legacy = (expert*rows + row)*stride
            narrow = expert*pitch + row*stride
            require(legacy==narrow and (narrow+vectors[-1]+15) < 150994944, 'bounded addresses')
for chunk in (0,1):
    actual = sorted((4*chunk+(lane+32*i)//18,(lane+32*i)%18) for lane in range(32) for i in range(3) if i<2 or lane<8)
    require(actual==[(k,v) for k in range(4*chunk,4*chunk+4) for v in range(18)], 'down rank/vector coverage')
print(json.dumps({'current_cuda_relinked_sha256':control,'cuda_candidate_sha256':sha('libggml-cuda.so'),'operator_sha256':sha('operator'),
 'current_server_sha256':server,'unchanged_attention_object_sha256':attention,'unchanged_gdn_object_sha256':gdn,
 'current_moe_object_sha256':sha('../resume-q2-moe-cohorts/occamy-fused.cu.o'),'new_moe_object_sha256':sha('occamy-fused.cu.o'),
 'objects_in_each_link':147,'preserved_device_functions':len(old),'preserved_instruction_words':sum(len(words(body)) for body in old.values()),
 'candidate_function':sym,'new_instruction_words':len(words(new[sym])),'candidate_registers':63,'static_shared_bytes':128,'dynamic_shared_bytes':39424,
 'stack_bytes':0,'spill_load_bytes':0,'spill_store_bytes':0,'required_blocks_per_sm':2,'sms':24,'threads_per_block':512,
 'loops':{'control':base_counts,'candidate':cand_counts},'expert_pitch_bytes':pitch,'routed_tensor_bytes':256*pitch,'maximum_vector_offset':256*pitch-16,
 'builder_image':'sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed','runtime_image':'sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d'},indent=2))
