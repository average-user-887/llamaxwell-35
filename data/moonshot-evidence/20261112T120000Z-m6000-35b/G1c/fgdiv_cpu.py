#!/usr/bin/env python3
"""FGDIV CPU step: first divergent fg token index, cap0 (A) vs cap1024 (B), from G1b units.jsonl. Run: python3 -I fgdiv_cpu.py <units.jsonl>"""
import sys, json, itertools, collections
U = [json.loads(l) for l in open(sys.argv[1])]
def first_div(x, y):
    for i, (a, b) in enumerate(zip(x, y)):
        if a != b: return i
    return None if len(x) == len(y) else min(len(x), len(y))
def key(u): return (u['kind'], u['identity_pass'])
G = collections.defaultdict(list)
for u in U:
    if u['fg'].get('ids'): G[(u['kind'], u['identity_pass'], u['arm'])].append(u)
print('group,n,distinct_streams,len')
for k, v in sorted(G.items()): print(k, len(v), len({tuple(u['fg']['ids']) for u in v}), sorted({len(u['fg']['ids']) for u in v}))
print('\nwithin-arm repeat first-divergence (same workload; for kind b the u-segment differs per unit so only kind a / identity are comparable)')
for k, v in sorted(G.items()):
    if k[0] == 'a' or k[1]:
        d = [first_div(v[0]['fg']['ids'], u['fg']['ids']) for u in v[1:]]
        print(k, d)
print('\ncap0 vs cap1024 first divergent fg token index')
for kind in 'ab':
    for ident in (False, True):
        A, B = G.get((kind, ident, 'A'), []), G.get((kind, ident, 'B'), [])
        if not A or not B: continue
        if kind == 'b' and not ident:
            # u segment = (cycle*4+pos)%16; match units sharing the segment
            segs = lambda L: {(u['cycle'] * 4 + u['pos']) % 16: u for u in L}
            sa, sb = segs(A), segs(B); res = {s: first_div(sa[s]['fg']['ids'], sb[s]['fg']['ids']) for s in sorted(set(sa) & set(sb))}
            print(kind, 'timed matched-by-segment', res)
        else:
            res = sorted({first_div(a['fg']['ids'], b['fg']['ids']) for a in A for b in B}, key=lambda x: (x is None, x))
            print(kind, 'identity' if ident else 'timed', 'distinct first-div over all A x B pairs:', res)
print('\nworkload a: identity pass (stream:false, arrival fixed 4 s) vs timed pass (arrival after 8 bg tokens), same arm')
for arm in 'AB':
    I, T = G.get(('a', True, arm)), G.get(('a', False, arm))
    if I and T: print('arm', arm, 'identity-vs-timed first-div', first_div(I[0]['fg']['ids'], T[0]['fg']['ids']))
print('\nNOTE workload b: u-segment = (cycle*4+pos)%16 and arms never share a segment in ABBA/BAAB order (A:0,3,5,6,8,11,13,14 B:1,2,4,7,9,10,12,15), so cap0-vs-cap1024 greedy divergence for b is NOT measured (prompts differ). Within-arm repeats of a segment (cycle 4 == cycle 0) are identical.')
