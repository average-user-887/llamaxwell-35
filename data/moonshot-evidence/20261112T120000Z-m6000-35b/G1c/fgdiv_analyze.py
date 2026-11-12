#!/usr/bin/env python3
"""python3 -I fgdiv_analyze.py <fgdiv_out>: KL(cap0||cap1024) per fg row, same arrival mode; plus within-arm timing effect and repeat noise. Scale: stock two-slot vs solo KL .00306 (C2)."""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from g1c_analyze import kl_rows
d = sys.argv[1]; L = lambda a, m, r: np.load('%s/%s_%s_%d.npy' % (d, a, m, r))
def st(name, x, y):
    if not (os.path.exists(x) and os.path.exists(y)): return
    kl, top = kl_rows(np.load(x), np.load(y)); print('%-34s mean=%.5f max=%.5f same-top=%.1f%% (row0 KL %.5f)' % (name, kl.mean(), kl.max(), 100 * top.mean(), kl[0]))
    return kl.mean()
out = []
for m in 'TI':
    for r in range(2): out.append(st('cap0 vs cap1024 mode %s rep %d' % (m, r), '%s/A_%s_%d.npy' % (d, m, r), '%s/B_%s_%d.npy' % (d, m, r)))
    for a in 'AB': st('%s repeat noise mode %s (rep0 vs rep1)' % (a, m), '%s/%s_%s_0.npy' % (d, a, m), '%s/%s_%s_1.npy' % (d, a, m))
for a in 'AB': st('%s timing effect (T vs I) rep0' % a, '%s/%s_T_0.npy' % (d, a), '%s/%s_I_0.npy' % (d, a))
v = [x for x in out if x is not None]
if v:
    mm = max(v); print('\nscale: stock two-slot vs solo KL .00306 (336 rows; deep .00506)')
    print('VERDICT: ' + ('within gate (<= +.0010 over repeat noise; compare lines above)' if mm <= 0.0010 else 'outside gate - reported, not judged') + ' [max cap0-vs-cap1024 mean KL %.5f]' % mm)
    print('Hypothesis (batch composition) check: if row0 KL >> rows 1..31 KL, divergence is prefill-arrival borne (chunking/co-batch during prefill); if rows 1..31 equal row0, it is decode-path. Pair with server_*.log batch lines (-lv 4).')
