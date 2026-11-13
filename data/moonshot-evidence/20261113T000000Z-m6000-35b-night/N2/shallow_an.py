import sys, re, os, numpy as np
Dd, Od = sys.argv[1], sys.argv[2]
def load(p, k):
    nv = int(re.search(r'n_vocab=(\d+)', open(p + '/log.txt').read()).group(1)); return np.fromfile('%s/s%d.logits' % (p, k), np.float32).reshape(-1, nv)
def lsm(x): x = x.astype(np.float64); x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
def pth(c, a): return '%s/%s_pair0' % (Dd, c) if a == 'BASE' else '%s/%s_%s' % (Od, c, a)
for ref, arm in (('BASE', 'A2'), ('BASE', 'Ap'), ('A2', 'Ap')):
    kls, tops, dm, bad = [], [], [], 0
    for c in ('e1', 'ea', 'eb', 'ec'):
        for k in (0, 1):
            try: a, b = lsm(load(pth(c, ref), k)), lsm(load(pth(c, arm), k))
            except Exception as e: bad += 1; continue
            if a.shape != b.shape or not np.isfinite(b).all(): bad += 1; continue
            kl = (np.exp(a) * (a - b)).sum(1); kls.append(kl); tops.append(a.argmax(1) == b.argmax(1)); dm.append(kl.mean())
    if not kls: print('%s vs %s: no data' % (arm, ref)); continue
    kl = np.concatenate(kls); st = np.concatenate(tops); n = len(dm); h = T.get(n - 1, 1.96) * np.std(dm, ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    print('%s vs %s: rows=%d units=%d missing=%d dmean=%+.5f CIunit=[%+.5f,%+.5f] top=%+.2fpt max=%.4f' % (arm, ref, len(kl), n, bad, kl.mean(), kl.mean() - h, kl.mean() + h, 100 * (st.mean() - 1), kl.max()))
