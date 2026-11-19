from pathlib import Path

parent = Path.cwd().parent
diag = (parent / 'last-ms-q8-gdn-diag/commands.sh').read_text().splitlines()
cohort = (parent / 'hundred-q2-cohort-layout/compile.sh').read_text().splitlines()
compile, = [l for l in diag if '/usr/local/cuda/bin/nvcc' in l]
link, = [l for l in cohort if l.startswith('/usr/bin/g++-14') and '/work/control-relinked.so' in l]
assert link.count('/q8/occamy-gdn.cu.o') == 1
candidate = link.replace('/work/control-relinked.so', '/work/libggml-cuda.so').replace(
    '/q8/occamy-gdn.cu.o', '/work/occamy-gdn.cu.o')
assert '-o /work/occamy-gdn.cu.o' in compile
commands = [
    '#!/usr/bin/env bash', 'set -euo pipefail', 'cd /b/ggml/src/ggml-cuda',
    compile, link, candidate,
    '/usr/local/cuda/bin/cuobjdump -sass /q8/occamy-gdn.cu.o > /work/control.sass',
    '/usr/local/cuda/bin/cuobjdump -sass /work/occamy-gdn.cu.o > /work/candidate.sass',
]
Path('compile.sh').write_text('\n'.join(commands) + '\n')
