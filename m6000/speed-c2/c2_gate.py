#!/usr/bin/env python3
"""C2 gate table from runs/results.json (valid runs only)."""
import json, statistics as st, sys
R = [r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/qa-speed-cards/c2/runs/results.json')) if not r['invalid'] and r.get('ok') and r.get('makespan_s')]
print('| run | arm | makespan s | p95 gap a0/a1 s | max gap a0/a1 s | void | tmax C | sha |\n|---|---|---|---|---|---|---|---|')
for r in R: print('| %s | %s | %.1f | %.3f/%.3f | %.2f/%.2f | %s | %s | %s |' % (r['run'], r['arm'], r['makespan_s'], r['p95_gap_a0'], r['p95_gap_a1'], r['max_gap_a0'], r['max_gap_a1'], r['n_void'], r['tmax'], (r.get('run_sha') or '')[:10]))
for c in ('96', '64'):
    off = [r for r in R if r['arm'] == 'OFF' + c]; on = [r for r in R if r['arm'] == 'ON' + c]
    if not off or not on: continue
    mo, mn = st.mean(x['makespan_s'] for x in off), st.mean(x['makespan_s'] for x in on)
    gain = (mo - mn) / mo * 100
    p = {a: (st.mean(x['p95_gap_a%d' % a] for x in off), st.mean(x['p95_gap_a%d' % a] for x in on)) for a in (0, 1)}
    print('\n%sK: OFF n=%d mean %.1f s | ON n=%d mean %.1f s | gain %+.2f%% | p95 stall a0 %.3f->%.3f (%+.1f%%) a1 %.3f->%.3f (%+.1f%%)' % (
        c, len(off), mo, len(on), mn, gain, p[0][0], p[0][1], (p[0][1] / p[0][0] - 1) * 100, p[1][0], p[1][1], (p[1][1] / p[1][0] - 1) * 100))
