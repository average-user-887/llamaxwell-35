import hashlib
import json
import re
from pathlib import Path

card = Path(__file__).resolve().parent
c = card.parent
baseline = "19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00"
new_kernel = "_Z26ofa_sparse_select_r2_top16P15ofa_sparse_planPKfPK15ofa_sparse_metai"

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def words(path):
    parts = re.split(r"Function : (\S+)\n", path.read_text())
    return {name: re.findall(r"/\* (0x[0-9a-f]{16}) \*/", body)
            for name, body in zip(parts[1::2], parts[2::2], strict=True)}

old, new = [words(card / f"{label}.sass") for label in ("control", "candidate")]
assert len(old) == 33 and all(w and w == new.get(n) for n, w in old.items())
assert new.keys() - old.keys() == {new_kernel} and new[new_kernel]
assert sha(card / "control-relinked.so") == baseline
assert sha(c / "resume-q2-moe-cohorts/libggml-cuda.so") == baseline
compile_log = (card / "build.log").read_text()
resources = re.search(
    rf"Function properties for {new_kernel}\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n"
    r"ptxas info\s*: Used (\d+) registers, used (\d+) barriers, (\d+) bytes smem", compile_log)
assert resources
stack, spill_store, spill_load, regs, barriers, shared = map(int, resources.groups())
assert stack == spill_store == spill_load == 0
assert 512 * ((regs + 7) // 8 * 8) <= 65536 and shared <= 49152
link_flags = (card / "compile.sh").read_text().splitlines()
assert link_flags[2].replace("control-relinked.so", "libggml-cuda.so").replace(
    "CMakeFiles/ggml-cuda.dir/fattn-occamy.cu.o", "/work/fattn-occamy.cu.o") == link_flags[4]
print(json.dumps({
    "control_sha256": baseline, "candidate_sha256": sha(card / "libggml-cuda.so"),
    "harness_sha256": sha(card / "test-backend-ops"),
    "q8_gdn_object_sha256": sha(c / "night-gdn-t2-q8-fast/occamy-gdn.cu.o"),
    "q2_moe_object_sha256": sha(c / "resume-q2-moe-cohorts/occamy-fused.cu.o"),
    "unchanged_attention_device_functions": len(old),
    "unchanged_attention_instruction_words": sum(map(len, old.values())),
    "added_device_function": new_kernel, "added_instruction_words": len(new[new_kernel]),
    "registers": regs, "barriers": barriers, "static_shared_bytes": shared, "stack_bytes": stack,
    "spill_load_bytes": spill_load, "spill_store_bytes": spill_store, "arch": "sm_52", "selector_cta_threads": 512,
    "known_80k_aligned_widths": list(range(80128, 83201, 256)),
}, indent=2))
