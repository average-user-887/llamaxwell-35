#!/usr/bin/env python3
"""Card-2 S4: simple paired BFCL comparison of two arm.json receipts (same logic as bfcl_pairs.paired_report counts,
without its identity/arm binding). usage: pairs.py OFF_arm.json ON_arm.json"""
import json, sys
a, b = (json.load(open(p))['samples'] for p in sys.argv[1:3])
ids = sorted(set(a) | set(b)); c = dict(both_pass=0, both_fail=0, off_only=0, on_only=0, unresolved=0)
for k in ids:
    x, y = a.get(k), b.get(k)
    if not x or not y or (x['status'] == 'infra' and x['score'] is None) or (y['status'] == 'infra' and y['score'] is None): c['unresolved'] += 1; continue
    c[{(1, 1): 'both_pass', (0, 0): 'both_fail', (0, 1): 'on_only', (1, 0): 'off_only'}[((x['score'] or 0), (y['score'] or 0))]] += 1
n = len(ids); off = c['both_pass'] + c['off_only']; on = c['both_pass'] + c['on_only']
print(json.dumps(dict(n=n, **c, off_pass=off, on_pass=on, discordant=c['off_only'] + c['on_only'],
                      rel_drop=(off - on) / off if off else None)))
