#!/usr/bin/env python3
"""Shallow set: KL(F16 stock pair0 || q8 arm) per case/seq from dx2seq dumps. usage: shallow_analyze.py ROUNDS_D1B_DUMPS Q8S_DUMPS"""
import sys, re, os, numpy as np
Dd, Qd = sys.argv[1], sys.argv[2]
def load(p, k):
    nv = int(re.search(r'n_vocab=(\d+)', open(p + '/log.txt').read()).group(1)); return np.fromfile('%s/s%d.logits' % (p, k), np.float32).reshape(-1, nv)
def lsm(x): x = x.astype(np.float64); x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
res = ['# Q8S shallow set (e1,ea,eb,ec: 2 seqs; baseline = stock F16 pair0 dumps, D1B lib, N2SPLIT=0)', '']
for arm in ('Q8KV', 'Q8K', 'Q8V'):
    kls, tops, dm = [], [], []
    for c in ('e1', 'ea', 'eb', 'ec'):
        p = '%s/%s_%s' % (Qd, c, arm)
        if not os.path.exists(p + '/s0.logits') or 'rc=0' not in open(p + '/log.txt').read(): continue
        for k in (0, 1):
            a, b = lsm(load('%s/%s_pair0' % (Dd, c), k)), lsm(load(p, k)); assert a.shape == b.shape and np.isfinite(b).all()
            kl = (np.exp(a) * (a - b)).sum(1); kls.append(kl); tops.append(a.argmax(1) == b.argmax(1)); dm.append(kl.mean())
    if not kls: res.append('- %s: not run' % arm); continue
    kl = np.concatenate(kls); st = np.concatenate(tops); n = len(dm); h = T.get(n - 1, 1.96) * np.std(dm, ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    d = kl.mean(); t = 100 * (st.mean() - 1); bad = d > .0010 or t < -.3
    res.append('- %s: rows=%d units=%d dmean=%+.5f CIunit=[%+.5f,%+.5f] top=%+.2fpt max=%.4f -> %s%s' % (arm, len(kl), n, d, d - h, d + h, t, kl.max(), 'FAIL' if bad else 'PASS', ' [MARGINAL]' if (not bad and (d > .0007 or t < -.2)) or d - h <= .0010 <= d + h or -.5 <= t <= -.1 else ''))
open(Qd + '/../SHALLOW_TABLE.md', 'w').write('\n'.join(res) + '\n'); print('\n'.join(res))
