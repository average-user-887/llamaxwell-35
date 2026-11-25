#!/usr/bin/env python3
"""Card-2 S3 compare: logprob-dump parity (proxy, NOT KLD). usage: s3_cmp.py REF ARM [ARM...]
Per request: decode steps aligned until first chosen-token divergence; top-1 agreement over aligned steps,
|dlogprob| of the reference's chosen token (when present in both top-8 lists), first divergence step."""
import re, sys, json
from pathlib import Path
OUT = Path("/tmp/qa-prefill-card/s3")
def reqs(p):
    out = []; cur = None
    for ln in open(p):
        m = re.match(r'N slot=\d+ n_dec=(\d+) rows=\d+ \[([^\]]*)\] ids=(\d+)', ln)
        if not m: continue
        n = int(m.group(1))
        if n == 0: cur = []; out.append(cur)
        top = {int(k): float(v) for k, v in (x.split('=') for x in m.group(2).split(','))}
        cur.append((n, top, int(m.group(3))))
    return out
ref, arms = sys.argv[1], sys.argv[2:]
R = {a: reqs(OUT / a / 'mv2.log') for a in [ref] + arms}
summ = {}
for a in arms:
    tot = agree = 0; dl = []; divs = []
    for ri, (x, y) in enumerate(zip(R[ref], R[a])):
        div = None
        for (n1, t1, i1), (n2, t2, i2) in zip(x, y):
            tot += 1; agree += (i1 == i2)
            if i1 in t1 and i1 in t2: dl.append(abs(t1[i1] - t2[i1]))
            if i1 != i2: div = n1; break
        divs.append(div)
    summ[a] = dict(ref=ref, nreq=(len(R[ref]), len(R[a])), steps=tot, top1_agree=agree, mean_abs_dlp=sum(dl) / max(1, len(dl)),
                   max_abs_dlp=max(dl) if dl else 0.0, n_dlp=len(dl), first_div_per_req=divs)
    print(a, json.dumps(summ[a]))
json.dump(summ, open(OUT / ('cmp_%s.json' % ref), 'w'), indent=1)
