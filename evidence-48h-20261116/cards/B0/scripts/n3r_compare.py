#!/usr/bin/env python3
"""usage: n3r_compare.py A_dir B_dir            -> per-turn token sha equality (A vs A check) and makespan.
       n3r_compare.py A1_dir A2_dir B_dir      -> floor style: floor = |A2 vs A1| spread (n=2 vs n=1) for makespan and per-agent decode time; B reported vs mean(A1,A2) and called WITHIN FLOOR or OUTSIDE.
Also prints void-turn counts per run (generated <64 tokens or >50% repeated 4grams): a run with void turns is not representative."""
import json, sys
D = sys.argv[1:]; assert len(D) in (2, 3), __doc__
R = [json.load(open(d + '/run.json')) for d in D]
k = lambda r: {(t['agent'], t['turn']): t for t in r['turns_rec']}
for d, r in zip(D, R): print('%s: makespan %.1f s, void turns %s/%d' % (d.rstrip('/').split('/')[-1], r['makespan_s'], r.get('n_void', 'n/a'), len(r['turns_rec'])))
def sheq(a, b):
    ka, kb = k(a), k(b); diff = 0
    for key in sorted(set(ka) | set(kb)):
        x, y = ka.get(key, {}).get('token_sha'), kb.get(key, {}).get('token_sha'); eq = x == y and x is not None; diff += not eq
        print(key, ka.get(key, {}).get('kind'), 'EQ' if eq else 'DIFF', x, y)
    return diff
if len(R) == 2:
    a, b = R; diff = sheq(a, b)
    print('makespan A %.1f B %.1f (%+.2f%%)  turns differing: %d  ->  %s' % (a['makespan_s'], b['makespan_s'], 100 * (b['makespan_s'] / a['makespan_s'] - 1), diff, 'PASS' if not diff else 'FAIL'))
    sys.exit(1 if diff else 0)
a1, a2, b = R
print('-- A1 vs A2 tokens'); d12 = sheq(a1, a2); print('-- A1 vs B tokens'); d1b = sheq(a1, b)
def dec(r, ag): return sum((t.get('predicted_ms') or 0) for t in r['turns_rec'] if t['agent'] == ag) / 1000
rows = [('makespan', a1['makespan_s'], a2['makespan_s'], b['makespan_s'])] + [('decode_s agent%d' % g, dec(a1, g), dec(a2, g), dec(b, g)) for g in (0, 1)]
out = 0
print('%-16s %8s %8s %8s | floor %% | B vs mean(A) %% | verdict' % ('metric', 'A1', 'A2', 'B'))
for n, x1, x2, y in rows:
    m = (x1 + x2) / 2; fl = 100 * abs(x2 - x1) / m if m else 0; dl = 100 * (y / m - 1) if m else 0; v = 'WITHIN FLOOR' if abs(dl) <= fl else ('B faster' if dl < 0 else 'B slower'); out += v == 'B slower'
    print('%-16s %8.1f %8.1f %8.1f | %6.2f | %+8.2f | %s' % (n, x1, x2, y, fl, dl, v))
print('token equality: A1==A2 turns differing %d, A1 vs B turns differing %d -> %s' % (d12, d1b, 'A-vs-A PASS' if not d12 else 'A-vs-A FAIL (floor invalid)'))
sys.exit(1 if d12 else 0)
