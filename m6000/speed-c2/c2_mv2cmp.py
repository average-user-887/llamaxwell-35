#!/usr/bin/env python3
"""C2 logprob parity PROXY (not FP64 KLD): MV2 top-8 dumps of two N3R runs (OFF ref, ON arm), per slot, per request
(segmented on n_dec reset), first verification row of each step keyed by n_dec, aligned until chosen ids diverge.
Reports top-1 agreement, mean/max |dlogprob| of ref top-1, and top-8 truncated KL(ref||arm). usage: c2_mv2cmp.py REF_LOG ARM_LOG"""
import re, sys, json, math
def load(p):
    S = {}
    for ln in open(p):
        m = re.match(r'([NV]) slot=(\d+) n_dec=(\d+).*? \[([^\]]*)\].*?ids=([\d,]+)', ln)
        if not m: continue
        s, n = int(m.group(2)), int(m.group(3)); top = {int(k): float(v) for k, v in (x.split('=') for x in m.group(4).split(','))}
        ids = [int(x) for x in m.group(5).split(',')]
        L = S.setdefault(s, [])
        if not L or n < L[-1][-1][0]: L.append([])
        L[-1].append((n, top, ids))
    return S
A, B = load(sys.argv[1]), load(sys.argv[2])
tot = agree = 0; dl = []; kl = []; divs = []
for s in sorted(set(A) & set(B)):
    for x, y in zip(A[s], B[s]):
        yb = {n: (t, i) for n, t, i in y}; div = None
        for n, t1, i1 in x:
            if n not in yb: div = n; break
            t2, i2 = yb[n]; tot += 1; a1 = max(t1, key=t1.get); a2 = max(t2, key=t2.get); agree += a1 == a2
            if a1 in t2: dl.append(abs(t1[a1] - t2[a1]))
            kl.append(sum(math.exp(v) * (v - t2[k]) for k, v in t1.items() if k in t2))
            if i1 != i2: div = n; break
        divs.append(div)
print(json.dumps(dict(proxy='MV2 top-8 logprob parity, not FP64 KLD', steps=tot, top1_agree=agree, mean_abs_dlp=sum(dl) / max(1, len(dl)), max_abs_dlp=max(dl) if dl else 0,
                      mean_trunc_kl=sum(kl) / max(1, len(kl)), max_trunc_kl=max(kl) if kl else 0, nreq=len(divs), first_div=divs)))
