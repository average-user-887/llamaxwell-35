import hashlib
import json
import re
from pathlib import Path

def read(path):
    seg = re.split(r'Function : (\S+)\n', Path(path).read_text())
    return dict(zip(seg[1::2],seg[2::2]))

def words(body):
    return re.findall(r'/\* (0x[0-9a-f]{16}) \*/',body)

old, new = read('old.sass'), read('new.sass')
orig = '_ZN10occamy_gdn10gdn_fused2ILi2EEEvNS_7Params2E'
clock = '_ZN10occamy_gdn16gdn_fused2_clockENS_7Params2EPVj'
assert new.keys() - old.keys() == {clock}
assert old.keys() - new.keys() == set()
assert all(words(new[n]) == words(v) and words(v) for n,v in old.items())
def points(op):
    return [int(m, 16) for m in re.findall(r'/\*([a-f0-9]{4})\*/[^\n]*'+op, new[clock])]
marks = points('SR_CLOCKLO')
spill = points(r'\b(?:LDL|STL)\b')
assert len(marks)==6 and len(set(marks))==6
assert all(p >= 0 for p in spill)
assert new[clock].count('BAR.SYNC') == old[orig].count('BAR.SYNC')
build = Path('build.log').read_text()
def resources(name):
    r=re.search("Function properties for "+re.escape(name)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (?:\d+ bytes cumulative stack size, )?(\d+) bytes smem',build)
    assert r, name
    return dict(zip(['stack','spill_store','spill_load','registers','barriers','static_shared'],map(int,r.groups())))
ref,test = resources(orig),resources(clock)
assert ref['registers']==test['registers']==80 and ref['barriers']==test['barriers']==1
print(json.dumps({'unchanged_device_functions':len(old),'new_device_functions':[clock],'original_t2':ref,'instrumented_t2':test,
  'dynamic_shared_bytes':27392,'static_bar_sync_instructions_in_each':new[clock].count('BAR.SYNC'),'clock_offsets_hex':[hex(p) for p in marks],
  'spill_offsets_hex':[hex(p) for p in spill],'spill_mark_intervals':[sum(p>m for m in marks) for p in spill],'whole_control_t2_instruction_sha256':hashlib.sha256(''.join(words(old[orig])).encode()).hexdigest()},indent=2))
