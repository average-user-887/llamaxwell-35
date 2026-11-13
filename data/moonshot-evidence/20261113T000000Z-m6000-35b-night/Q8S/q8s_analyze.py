#!/usr/bin/env python3
"""Q8S analyzer. usage: python3 -I q8s_analyze.py OUT  (OUT/A -> arm A vectors, OUT/Q8*/ arm vectors). Reuses NC2 hardened loader/KL (g1c_analyze_q8.py, unchanged copy)."""
import sys, os, importlib.util, glob
here = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('ga', here + '/g1c_analyze_q8.py'); ga = importlib.util.module_from_spec(sp); sp.loader.exec_module(ga)
out = sys.argv[1]; arms = [a for a in ('Q8KV', 'Q8K', 'Q8V') if glob.glob('%s/%s/doc*_b*.f32' % (out, a))]
md = ['# Q8S long set: KL(F16 arm A || q8_0 arm) per bucket; gate dmean<=+.0010 and same-top>=-.3 pt', '']
for arm in arms:
    md += ['## ' + arm]
    for B in ga.BUCKETS:
        r = ga.pair(out, 'A', arm, B)
        if r is None: md.append('- %d: n/a' % B); continue
        bad = r['dmean'] > .0010 or r['top_delta_pt'] < -.3
        marg = (not bad and (r['dmean'] > .0007 or r['top_delta_pt'] < -.2)) or (r['ci_doc'][0] <= .0010 <= r['ci_doc'][1]) or (-.5 <= r['top_delta_pt'] <= -.1)
        md.append('- %d: %s %s%s' % (B, 'FAIL' if bad else 'PASS', ga.fmt(r), ' [MARGINAL]' if marg else ''))
    md.append('')
open(out + '/Q8S_TABLE.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
