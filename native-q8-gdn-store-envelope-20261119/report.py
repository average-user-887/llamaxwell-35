import json
from math import isfinite
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
log = (root / 'operator.log').read_text()
assert log.count('GDN_ENVELOPE_OK rollback_intermediate_deliberately_missing') == 1
residency = re.findall(
    r'^GDN_RESIDENCY mode=(control|elided) sm=52 sms=24 active=3 grid=72 threads=256 registers=(\d+) local=0 shared=(\d+)$',
    log, re.M)
assert residency == [('control', '80', '27392'), ('elided', '80', '27392')], residency
scopes = re.findall(r'^GDN_COMPARE_SCOPE fixture=(\d+) mode=(\d+)$', log, re.M)
assert scopes == [(str(i), str(m)) for i in range(4) for m in (1, 2)], scopes
numeric = re.findall(r'^GDN_COMPARE fixture=(\d+) field=(\w+) words=(\d+) finite=(\d+) differing=(\d+)$', log, re.M)
expected = []
for fixture in range(4):
    for snapshot in ('intermediate_state', 'missing_intermediate'):
        for field, size in (('residual_both', 4096), ('convolution_both', 49152), ('final_state', 524288), (snapshot, 524288)):
            expected.append((str(fixture), field, str(size), str(size), '0'))
assert numeric == expected, numeric
assert re.findall(r'^GDN_CONTROL_SNAPSHOT fixture=(\d) finite_changed=524288 expected=524288$', log, re.M) == list('0123')
intervals = re.findall(r'^GDN_INTERVAL order=(\d+) mode=(\w+) reps=128 mean_us=([\d.]+)$', log, re.M)
expected_modes = 'original full elided elided full original elided full original original full elided'.split()
assert [mode for _, mode, _ in intervals] == expected_modes
assert [int(order) for order, _, _ in intervals] == list(range(12))
times = [float(duration) for _, _, duration in intervals]
assert all(isfinite(value) and value > 0 for value in times)
blocks = [dict(zip(expected_modes[i:i+3], times[i:i+3])) for i in range(0, len(times), 3)]
same_binary = [row['full'] - row['elided'] for row in blocks]
original = [row['original'] - row['elided'] for row in blocks]
overhead = [row['full'] - row['original'] for row in blocks]
floor = min(same_binary + original)
print(json.dumps({'technical_pass': True, 'rollback_intermediate_deliberately_missing': True,
                  'scope': 'synthetic full GDN kernel, distinct private buffers, four fixtures, no model dispatch',
                  'resident_ctas_per_sm': 3, 'grid': 72, 'intervals_us': blocks,
                  'same_binary_saved_us_per_layer': same_binary,
                  'original_control_saved_us_per_layer': original,
                  'predicate_control_overhead_us_per_layer': overhead,
                  'conservative_observed_min_us_per_layer': floor,
                  'necessary_solo_23us': floor >= 23, 'necessary_companion_6_4us': floor > 6.4}, indent=2))
