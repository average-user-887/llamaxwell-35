import sys, glob, os, csv, numpy as np, collections
out = sys.argv[1]
def lsm(a):
    a = a.astype(np.float64); m = a.max(); return a - (m + np.log(np.exp(a - m).sum()))
def load(arm):
    r = {}
    for f in glob.glob('%s/%s/*_r*_*.npy' % (out, arm)):
        b = os.path.basename(f)[:-4]; doc, rep, cp = b.split('_'); r[(doc, int(rep[1:]), int(cp))] = np.load(f)
    return r
def rows(arm):
    t = collections.defaultdict(dict)
    p = '%s/%s/rows.csv' % (out, arm)
    if os.path.exists(p):
        for x in csv.DictReader(open(p)): t[(x['doc'], int(x['rep']), int(x['checkpoint']))][int(x['row'])] = int(x['raw_top1'])
    return t
def nkv(p):
    last = 0; cps = (16384, 32768, p) if p > 65600 else (16384, 32768, 65536)
    for cp in cps:
        while cp - last > 16384: last += 16384
        if cp == p: break
        last = cp + 15
    e = p - 4; n = (e - last) % 1024 or 1024
    if n < 64: e -= n
    return (-(-e // 256) * 256) % 4096
def cmp(ref, tst):
    lr, lt = lsm(ref), lsm(tst); pr = np.exp(lr)
    return float((pr * (lr - lt)).sum()), float(0.5 * np.abs(pr - np.exp(lt)).sum()), int(ref.argmax() == tst.argmax())
arms = {a: load(a) for a in ['A', 'B', 'C', 'Ap', 'A2', 'B2']}
R = {a: rows(a) for a in arms}
L = []; csvo = open(out + '/G1_windows.csv', 'w'); csvo.write('pair,doc,checkpoint,nkv_mod4096,KL,TV,same_top1_row0,same_top1_16rows\n')
def pair(x, y, keys):
    res = collections.defaultdict(list)
    for k in sorted(keys):
        if k in arms[x] and k in arms[y] and k[1] == 0:
            kl, tv, s = cmp(arms[x][k], arms[y][k])
            a, b = R[x].get(k, {}), R[y].get(k, {}); com = [i for i in a if i in b]
            s16 = np.mean([a[i] == b[i] for i in com]) if com else float('nan')
            csvo.write('%s-vs-%s,%s,%d,%d,%.6g,%.6g,%d,%.4f\n' % (x, y, k[0], k[2], nkv(k[2]), kl, tv, s, s16))
            res[k[2]].append((kl, tv, s, s16, k[0]))
    return res
allk = set().union(*[set(v) for v in arms.values()])
T = {}
for x, y in [('C', 'A'), ('C', 'B'), ('A', 'B'), ('A', 'Ap'), ('A', 'A2'), ('B', 'B2')]:
    T[(x, y)] = pair(x, y, allk)
md = ['# G1 table (KL(ref||test) nats; ref = first-named; per-window = one full-vocab position; windows are (doc, checkpoint))', '']
md.append('| bucket | pair | n win | meanKL | maxKL | same-top1 row0 | same-top1 16 rows | meanTV |'); md.append('|---|---|---|---|---|---|---|---|')
for cp in (16384, 32768, 65536, 65836):
    for pr, res in T.items():
        v = res.get(cp)
        if not v: continue
        kl = [a[0] for a in v]
        md.append('| %s (nkv%%4096=%d) | %s-vs-%s | %d | %.3g | %.3g | %.1f%% | %.1f%% | %.4f |' % ('shift65836' if cp == 65836 else str(cp), nkv(cp), pr[0], pr[1], len(v), np.mean(kl), np.max(kl), 100 * np.mean([a[2] for a in v]), 100 * np.nanmean([a[3] for a in v]), np.mean([a[1] for a in v])))
md.append(''); md.append('dmean = meanKL(B-vs-C) - meanKL(A-vs-C) on matched windows; top delta = same-top(B,C) - same-top(A,C) in pt (row0 / 16 rows)')
for cp in (16384, 32768, 65536, 65836):
    a = {x[4]: x for x in T[('C', 'A')].get(cp, [])}; b = {x[4]: x for x in T[('C', 'B')].get(cp, [])}; m = [d for d in a if d in b]
    if m: md.append('- %s: matched n=%d dmean=%+.5f top delta=%+.1f / %+.1f pt' % (cp, len(m), np.mean([b[d][0] - a[d][0] for d in m]), 100 * np.mean([b[d][2] - a[d][2] for d in m]), 100 * np.mean([b[d][3] - a[d][3] for d in m])))

md.append(''); md.append('Controls on codex window (rep0), full-vocab max abs logprob diff / KL: A-vs-A2 (OFF/OFF), B-vs-B2 (ON/ON)')
for x, y in (('A', 'A2'), ('B', 'B2')):
    for cp in (16384, 32768, 65536):
        k = ('codex', 0, cp)
        if k in arms[x] and k in arms[y]:
            md.append('- %s-vs-%s %d: maxabs %.3g KL %.3g' % (x, y, cp, float(np.nanmax(np.abs(arms[x][k].astype(np.float64) - arms[y][k]))), cmp(arms[x][k], arms[y][k])[0]))
md.append('Per-window values (full per-position distribution) in G1_windows.csv')
open(out + '/G1_TABLE.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md[-9:]))
