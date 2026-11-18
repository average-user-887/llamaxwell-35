import hashlib,json
from pathlib import Path
from report import candidate_numeric,pairs
p=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
for k,v in json.loads((p/'source-pins.json').read_text()).items():assert h((p/k).read_bytes())==v['portable_sha256']
d=json.loads((p/'operator.receipt.json').read_text())
r=d['complete_measurement']
assert r['technical_pass'] and not r['materiality_pass'] and d['status']['runner_exit']==d['status']['cleanup_exit']==0
numeric=candidate_numeric('\n'.join(d['check']['observed_lines'])); timing=pairs('\n'.join(d['perf']['observed_lines']))
assert json.loads(json.dumps(numeric))==r['numeric']['xmad'] and json.loads(json.dumps(timing))==r['timing']['timing']
assert all(not v['materiality_pass'] and max(v['incremental_savings_us'])<50 for v in timing.values())
print("Both measured full-output and Phase B comparisons, routes and complete physical timing intervals independently reproduced. Materiality missed.")
