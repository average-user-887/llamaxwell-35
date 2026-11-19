from pathlib import Path
import hashlib
import difflib

here = Path(__file__).resolve().parent
source = here.parent / 'night-gdn-t2-q8-fast/occamy-gdn.cu'
base = source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'cdab6a823ca92a648f83dadf35379151347bdafac564949d0c759d36827728e9'
start = base.index('template <int T2>\n__global__ void __launch_bounds__(NT, 3) gdn_fused2(')
end = base.index('\nstruct DevState {', start)
control = base[start:end]
treatment = control.replace(
    'template <int T2>\n__global__ void __launch_bounds__(NT, 3) gdn_fused2(const Params2 PP) {',
    '__global__ void __launch_bounds__(NT, 3) gdn_store_envelope(const Params2 PP) {\n    constexpr int T2 = 2;', 1)
store = '                so1[j * SD / 4] = a;'
assert treatment.count(store) == 1
treatment = treatment.replace(store, '', 1)
out = '#include <numeric>\n#include <functional>\n#include <cmath>\n' + base[:end] + '\n' + treatment + '\n' + (here / 'envelope.inc').read_text() + base[end:]
(here/'occamy-gdn.cu').write_text(out)
(here/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True), out.splitlines(True),
    fromfile='a/occamy-gdn.cu', tofile='b/occamy-gdn.cu')))
