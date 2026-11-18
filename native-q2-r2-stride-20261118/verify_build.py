"""Verify linked inputs, retained original SASS, and both specialized loops."""
import collections
import hashlib
import json
import re
from pathlib import Path

card = Path(__file__).resolve().parent
c = card.parent

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def functions(path):
    parts = re.split(r"Function : (\S+)\n",path.read_text())
    return dict(zip(parts[1::2], parts[2::2],strict=True))

def words(text):
    return re.findall(r"/\* (0x[0-9a-f]{16}) \*/",text)

def loop(body):
    # Longest backwards branch is the 4-key retained-attention main loop.
    branches = [(int(m[1],16),int(m[2],16)) for m in re.finditer(
        r"/\*([0-9a-f]+)\*/\s+(?:@!?P\d\s+)?BRA (0x[0-9a-f]+)",body)]
    hi,lo = max(branches,key=lambda item:item[0]-item[1])
    inst = [(int(addr,16),op,operand) for addr,op,operand in re.findall(
        r"/\*([0-9a-f]+)\*/\s+(?:@!?P\d\s+)?([A-Z0-9]+)([^;\n]*);",body)]
    items = [(op,operand) for addr,op,operand in inst if lo<=addr<=hi]
    counts = collections.Counter(op for op,_ in items)
    ldg128 = [operand for op,operand in items if op == "LDG" and ".128" in operand]
    return {"pc_from":hex(lo),"pc_to":hex(hi),"instructions":len(items),"opcodes":dict(sorted(counts.items())),
            "vector_loads":ldg128}

baseline = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"
probe_sha = "58bab12a23f562ab7a455999c139fd6a162f6c05d63bda3c383c95afe24fcd36"
assert sha(card/"control-relinked.so") == sha(c/"last-ms-r2-selector/libggml-cuda.so") == baseline
assert sha(c/"last-ms-r2-probe/libggml-cuda.so") == probe_sha
src,old,new = (functions(p) for p in (card/"control.sass", c/"last-ms-r2-probe/candidate.sass",card/"candidate.sass"))
assert len(src)==34 and len(old)==35 and len(new)==37
assert all(words(v) and words(v)==words(new[k]) for d in (src,old) for k,v in d.items())
retained = next(v for k,v in old.items() if "ofa_sparse_attentionILi2ELi4" in k)
before = loop(retained)
names = sorted(new.keys()-old.keys())
assert len(names)==2 and all("ofa_sparse_attention_q2_stride" in n for n in names)
resources = {}
compile_log = (card/"build.log").read_text()
for name in names:
    pitch = int(re.search(r"ILj(\d+)",name)[1])
    r = re.search(rf"Function properties for {name}\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n"
                  r"ptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem",compile_log)
    assert r
    stack, stores, loads, regs, barriers, shared = map(int,r.groups())
    main = loop(new[name])
    assert stack == stores == loads == 0
    assert regs<=128 and shared <=49152 and len(main["vector_loads"])==8
    offsets = [int(m[1],16) if (m:=re.search(r"\+0x([0-9a-f]+)\]",x)) else 0 for x in main["vector_loads"]]
    assert sorted(offsets)==sorted([j*pitch for j in range(4)]*2), offsets
    for op in ("F2F","FFMA","FMUL","FADD","MUFU","FMNMX"):
        assert main["opcodes"][op]==before["opcodes"][op],op
    assert main["opcodes"]["SHFL"]==before["opcodes"]["SHFL"]+1
    assert main["instructions"]<before["instructions"] and main["opcodes"].get("XMAD",0)<before["opcodes"]["XMAD"]
    resources[str(pitch)] = dict(registers=regs,shared_bytes=shared,spill_loads=loads,spill_stores=stores,
                                 stack_bytes=stack,barriers=barriers,main_loop=main)
commands=(card/"compile.sh").read_text().splitlines()
assert commands[2].replace("control-relinked.so","libggml-cuda.so").replace("/fast/fattn-occamy.cu.o","/work/fattn-occamy.cu.o")==commands[4]
print(json.dumps(dict(control_sha256=baseline,probe_parent_sha256=probe_sha,candidate_sha256=sha(card/"libggml-cuda.so"),
    harness_sha256=sha(card/"test-backend-ops"), unchanged_selector_functions=len(src),
    unchanged_probe_parent_functions=len(old), unchanged_probe_parent_words=sum(len(words(b)) for b in old.values()),
    q2_moe_object_sha256=sha(c/"resume-q2-moe-cohorts/occamy-fused.cu.o"),q8_gdn_object_sha256=sha(c/"night-gdn-t2-q8-fast/occamy-gdn.cu.o"),
    baseline_loop=before, offsets=resources,active_widths=list(range(80128,83201,256))),indent=2))
