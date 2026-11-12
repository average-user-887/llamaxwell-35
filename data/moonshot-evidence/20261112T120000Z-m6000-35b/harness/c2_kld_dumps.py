#!/usr/bin/env python3
# C2 (NEW file). CPU-only recompute from existing dx2seq logits dumps (float32 [rows, n_vocab]).
# ref   = D1B  <case>_pair0  (stock n=2 path, N2SPLIT=0)         lib: cand/D1B
# solo  = D1B  <case>_solo                                         (single-token decodes)
# arm   = D1B  <case>_pair1  (N2SPLIT=1). N2GDN/ATT1 add-ons are bit-equal to this in battery.log/G1 text; D1BF logits were NOT retained (only log.txt in dumps/*_g1).
import sys, re, json, csv, numpy as np
D = sys.argv[1]; OUT = sys.argv[2]
def load(p):
    nv = int(re.search(r'n_vocab=(\d+)', open(p + '/log.txt').read()).group(1))
    return {k: np.fromfile(f'{p}/s{k}.logits', np.float32).reshape(-1, nv) for k in (0, 1) if __import__('os').path.getsize(f'{p}/s{k}.logits')}
def lsm(x): x = x.astype(np.float64); x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
def kl(a, b):  # KL(softmax a || softmax b) per row
    la, lb = lsm(a), lsm(b); return (np.exp(la) * (la - lb)).sum(1)
cases = ['e1', 'ea', 'eb', 'ec', 'ed']
bucket = {'e1': '3K+3K', 'ea': '3K+3K reorder', 'eb': '3K+3K cancel', 'ec': '3K+6K reuse/restart', 'ed': '98K+90K restored slots'}
rows = []; fails = []
for c in cases:
    R = D + '/rounds'
    ref = load(f'{R}/D1B/dumps/{c}_pair0'); solo = load(f'{R}/D1B/dumps/{c}_solo'); p1 = load(f'{R}/D1B/dumps/{c}_pair1'); arm = p1
    for k in (0, 1):
        r, s, a, q = ref[k], solo[k], arm[k], p1[k]
        assert r.shape == s.shape == a.shape == q.shape, (c, k, r.shape, a.shape)
        for name, x, y in (('arm_vs_stockpair0', r, a), ('stockpair0_vs_solo', s, r), ('arm_vs_solo', s, a), ('n2split_only_vs_solo', s, q)):
            kk = kl(x, y); t = (x.argmax(1) == y.argmax(1)); be = (x.view(np.uint32) == y.view(np.uint32)).all(1)
            rows.append(dict(case=c, bucket=bucket[c], seq=k, cmp=name, rows=len(kk), meanKL=float(kk.mean()), maxKL=float(kk.max()), same_top1=float(t.mean()),
                             bitexact_rows=int(be.sum()), maxabs=float(np.abs(x - y).max())))
            if name in ('arm_vs_stockpair0', 'arm_vs_solo', 'stockpair0_vs_solo'):
                for i in np.where(~t)[0]: fails.append(dict(case=c, seq=k, step=int(i), cmp=name, ref_top=int(x[i].argmax()), arm_top=int(y[i].argmax()), KL=float(kk[i])))
w = csv.DictWriter(open(OUT + '/C2.csv', 'w'), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
w = csv.DictWriter(open(OUT + '/C2_top1_failures.csv', 'w'), fieldnames=['case', 'seq', 'step', 'cmp', 'ref_top', 'arm_top', 'KL']); w.writeheader(); w.writerows(fails)
agg = {}
for name in ('arm_vs_stockpair0', 'stockpair0_vs_solo', 'arm_vs_solo', 'n2split_only_vs_solo'):
    for sel, f in (('all5cases', lambda r: True), ('short_<=6K', lambda r: r['case'] != 'ed'), ('deep_98K+90K', lambda r: r['case'] == 'ed')):
        R = [r for r in rows if r['cmp'] == name and f(r)]; n = sum(r['rows'] for r in R)
        agg[f'{name}|{sel}'] = dict(rows=n, meanKL=sum(r['meanKL'] * r['rows'] for r in R) / n, same_top1=sum(r['same_top1'] * r['rows'] for r in R) / n,
                                    bitexact_rows=sum(r['bitexact_rows'] for r in R), maxKL=max(r['maxKL'] for r in R))
json.dump(agg, open(OUT + '/C2_agg.json', 'w'), indent=1); print(json.dumps(agg, indent=1)); print(len(fails), 'top1 failures')
