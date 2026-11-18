"""Replay both physical stage reports entirely from public typed evidence."""
import hashlib,json
from pathlib import Path
from report import numeric,timing
p=Path(__file__).resolve().parent
read=lambda n:json.loads((p/n).read_text())
for name,record in read('source-pins.json').items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==record['portable_sha256']
d=read('operator.receipt.json')
assert d['build']==read('build.json') and d['status']['runner_exit']==d['status']['cleanup_exit']==0
for stage,fn in [('check',numeric),('perf',timing)]:
    raw='\n'.join(d[stage]['observed_lines'])+'\n'
    obs=json.loads(json.dumps(fn(raw)))
    assert obs==d[stage]['validated']==d['complete_measurement'][{'check':'numeric','perf':'timing'}[stage]]
r=d['complete_measurement']
assert r['technical_pass'] and not r['materiality_pass']
assert all(0<s<50 for t in r['timing']['results'].values() for s in t['n2split_minus_cohorts_us'])
print('All native controls, eight-expert routes, bitwise results, physical callbacks, twelve intervals and the recorded materiality miss reproduced.')
