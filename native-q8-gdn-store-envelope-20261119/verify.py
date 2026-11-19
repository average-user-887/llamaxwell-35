from collections import Counter
from pathlib import Path
import hashlib
import json
import re

def functions(file):
    out = {}
    for part in re.split(r'Function : ', Path(file).read_text())[1:]:
        name = part.splitlines()[0].strip()
        out[name] = re.findall(r'/\* (0x[0-9a-f]+) \*/', part)
    return out

control, candidate = (functions(s) for s in ('control.sass', 'candidate.sass'))
assert len(control) == 17, control.keys()
assert all(code and candidate.get(name) == code for name, code in control.items()), [
    name for name, code in control.items() if candidate.get(name) != code]
added, = candidate.keys() - control.keys()
assert 'gdn_store_envelope' in added
assembly = Path('candidate.sass').read_text()
treatment = assembly.split('Function : ' + added)[1].split('Function : ')[0]
assert not re.search(r'\b(?:LDL|STL)\b', treatment)
ptxas = Path('build.log').read_text()
lines = ptxas.splitlines()
start, = [i for i, line in enumerate(lines) if 'Compiling entry function' in line and 'gdn_store_envelope' in line]
window = '\n'.join(lines[start:start + 6])
assert '0 bytes spill stores, 0 bytes spill loads' in window, window
registers = int(re.search(r'Used (\d+) registers', window).group(1))
assert registers <= 80, window
sha = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
assert sha('control-relinked.so').startswith('369c6b')
print(json.dumps({'control_sha256': sha('control-relinked.so'),
                  'candidate_sha256': sha('libggml-cuda.so'),
                  'patch_sha256': sha('candidate.patch'),
                  'original_functions_unchanged': len(control),
                  'original_machine_words_unchanged': sum(map(len, control.values())),
                  'new_function': added, 'candidate_registers': registers, 'candidate_spills': 0,
                  'candidate_instruction_summary': dict(Counter(re.findall(r'/\*[0-9a-f]+\*/\s+([A-Z0-9_.]+)', treatment)))}, indent=2))
