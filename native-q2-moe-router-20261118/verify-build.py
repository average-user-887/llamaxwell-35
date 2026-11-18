import hashlib
import json
import re
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def instructions(file):
    parts = re.split(r'Function : (\S+)\n', Path(file).read_text())
    return {name: re.findall(r'/\* (0x[0-9a-f]{16}) \*/', body)
            for name, body in zip(parts[1::2], parts[2::2])}

old, new = instructions('control.sass'), instructions('candidate.sass')
added, = new.keys() - old.keys()
assert 'moe_q2_router' in added
assert len(old) == 17 and len(new) == 18
assert all(words and new[k] == words for k, words in old.items())
props = re.search(r'Function properties for '+added+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem', Path('build.log').read_text())
assert props and props.groups()[:3] == ('0','0','0') and int(props[4]) <= 64 and props.groups()[4:] == ('1','96')
control = sha('control-relinked.so')
assert control == sha('../last-ms-r2-selector/libggml-cuda.so') == '9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e'
gdn = sha('../night-gdn-t2-q8-fast/occamy-gdn.cu.o')
assert gdn == 'bcd063306ea65f5e8143a8d6501960dbfeda359f5b3a261d5d91d612c786eac0'
link1,link2 = [line.split() for line in Path('compile.sh').read_text().splitlines() if line.startswith('/usr/bin/g++-14')]
assert link1.count('/fast/fattn-occamy.cu.o')==1 and sha('../last-ms-r2-selector/fattn-occamy.cu.o')=='c210591d286949d6e5bbdfd701d2fc6b657eabba378226ddce0c07a344fa3a0f'
assert len(link1)==len(link2) and link1.count('/q8/occamy-gdn.cu.o')==1
assert sum(x.endswith('.o') for x in link1)==147
assert [(a,b) for a,b in zip(link1,link2) if a!=b] == [
    ('/work/control-relinked.so','/work/libggml-cuda.so'),
    ('/control/occamy-fused.cu.o','/work/occamy-fused.cu.o')]
print(json.dumps(dict(
    native_control_library_sha256=control, candidate_library_sha256=sha('libggml-cuda.so'),
    operator_sha256=sha('operator'), candidate_object_sha256=sha('occamy-fused.cu.o'),
    control_object_sha256=sha('../resume-q2-moe-cohorts/occamy-fused.cu.o'),
    original_q8_gdn_object_sha256=gdn, linked_objects=147,
    builder_image='sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed',
    runner_image='sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d',
    preserved_device_functions=len(old), preserved_instruction_words=sum(map(len,old.values())),
    added_device_function=added, added_instruction_words=len(new[added]), registers=int(props[4]),
    static_shared_bytes=96,dynamic_shared_bytes=39424,stack_bytes=0,spill_load_bytes=0,spill_store_bytes=0,
    cc='sm_52',threads_per_block=512, blocks_per_grid=48,required_blocks_per_sm=2),indent=2))
