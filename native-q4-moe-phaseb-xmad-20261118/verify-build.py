"""Bind the CUDA object and link to the tested combined width-four baseline."""
import hashlib, json, re
from pathlib import Path
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(path):
    parts=re.split(r'Function : (\S+)\n',Path(path).read_text())
    return dict(zip(parts[1::2],parts[2::2]))
def words(s):
    return re.findall(r'/\* (0x[0-9a-f]{16}) \*/',s)
old,new=functions('control.sass'),functions('candidate.sass')
fresh=set(new)-set(old)
assert len(old)==18 and len(fresh)==1
symbol,=fresh
assert 'moe_q4_phaseb_xmad' in symbol
for key,body in old.items():
    assert words(body) and words(body)==words(new[key]),key
body=new[symbol]
ins=re.findall(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([^;]+?)\s*;',body)
signed=[int(addr,16) for addr,op in ins if op.startswith('XMAD.S16.S16')]
assert len(signed)>=128 and not re.search(r'\b(?:LDL|STL)\b',body)
m=re.search('Function properties for '+symbol+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem',Path('build.log').read_text())
assert m and m.groups()[:3]==('0','0','0') and int(m[4])<=64 and int(m[6])==96
assert digest('control-relinked.so')==digest('../resume-q4-combined-native/libggml-cuda.so')=='abe7dcfdbf2d7df0b31b5478212f3234ffdd8e2b8f9a077c0a216b2e32865aae'
report={'control':digest('control-relinked.so'),'candidate':digest('libggml-cuda.so'),'operator':digest('operator'),
        'existing_device_functions_unchanged':len(old),'existing_instruction_words_unchanged':sum(len(words(x)) for x in old.values()),
        'candidate_symbol':symbol,'signed_xmad_static_count':len(signed),'first_signed_xmad_offset':hex(min(signed)),
        'last_signed_xmad_offset':hex(max(signed)),'registers':int(m[4]),'static_shared_bytes':int(m[6]),
        'dynamic_shared_bytes':39424,'threads_per_cta':512,'ctas':48,'stack':0,'spill_stores':0,'spill_loads':0,
        'resident_admission':'actual SKU + at least two simultaneously active CTAs per SM; CPU did not query physical GPU'}
print(json.dumps(report,indent=2))
