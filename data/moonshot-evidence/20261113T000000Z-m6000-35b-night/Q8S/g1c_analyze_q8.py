#!/usr/bin/env python3
"""G1c analyzer. usage: python3 -I g1c_analyze.py <OUT> [--selftest <G1 arm dir root>]
Reads <OUT>/<ARM>/doc<k>_b<B>.f32 (float32 [64, n_vocab] logits), ARM in A B Ap C.
Per bucket: dmean = mean KL(A||X) over rows, same-top delta (pt) = 100*(P[argmax equal]-1), max, 95% CI of the mean
(ci_row: t-interval over rows; ci_doc: t-interval over per-doc means, the conservative one).
Gate (B vs A): PASS iff dmean<=+.0010 and same-top delta>=-0.3 pt in EVERY bucket. Never relaxed.
Noise floors on the same rows: A vs Ap, A vs C (C subset if C did not reach a bucket). Also C-relative BUNDLE3-style dmean = KL(C||B)-KL(C||A), paired per row."""
import sys, os, glob, json, re
import numpy as np
BUCKETS = (32768, 65536, 98304); NR = int(os.environ.get('G1C_NR', 64))
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
def tcrit(n): return T975.get(n, 1.96) if n > 1 else float('nan')
def lsm(x):
    x = x.astype(np.float64); m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
def kl_rows(ref, test):
    """KL(ref||test) per row and same-top flags; ref/test float logits [n, nv]"""
    p, q = lsm(ref), lsm(test); kl = (np.exp(p) * (p - q)).sum(-1)
    return kl, ref.argmax(-1) == test.argmax(-1)
def ci(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return (float('nan'), float('nan'))
    h = tcrit(n - 1) * v.std(ddof=1) / np.sqrt(n); return (v.mean() - h, v.mean() + h)
class AnalyzerError(Exception):
    """Raised for partial runs, wrong dims or non-finite values; never swallowed."""
NV = int(os.environ.get('G1C_NV', 248320))   # expected vocab size (Qwen3.5 family, n_vocab in g1c_rows log)
def expected_docs():
    return [int(x) for x in os.environ.get('G1C_DOCS', '0,1,2,3,4,5,6,7').split(',')]
REQUIRED_ARMS = ('A', 'B')   # every expected doc x bucket must exist; Ap/C are deliberate subsets but any file present is fully validated
def load(out, arm, k, B):
    """Strict load: returns float32 [NR, NV] or None if the file is absent. Wrong size / NaN / Inf raise AnalyzerError."""
    f = '%s/%s/doc%d_b%d.f32' % (out, arm, k, B)
    if not os.path.exists(f): return None
    sz = os.path.getsize(f); want = NR * NV * 4
    if sz != want:
        raise AnalyzerError('%s: size %d bytes, expected %d (%d rows x vocab %d x 4); got %s rows at that vocab%s' % (f, sz, want, NR, NV, ('%.3f' % (sz / 4 / NV)),
                            ' (not a multiple of vocab: wrong vocab size / truncated)' if sz % (4 * NV) else ' (partial or wrong row count)'))
    a = np.fromfile(f, dtype=np.float32).reshape(NR, NV)
    bad = ~np.isfinite(a)
    if bad.any():
        r = np.argwhere(bad); raise AnalyzerError('%s: %d non-finite values (NaN/Inf), first at row %d col %d' % (f, bad.sum(), r[0][0], r[0][1]))
    return a
def check_complete(out, buckets=BUCKETS):
    """Reject partial runs: required arms must have every expected doc x bucket."""
    miss = []
    for arm in REQUIRED_ARMS:
        for k in expected_docs():
            for B in buckets:
                f = '%s/%s/doc%d_b%d.f32' % (out, arm, k, B)
                if not os.path.exists(f) or os.path.getsize(f) == 0: miss.append(f.replace(out + '/', ''))
    if miss: raise AnalyzerError('partial run: %d required files missing/empty (arms %s, docs %s): %s%s' % (len(miss), ','.join(REQUIRED_ARMS), expected_docs(), ', '.join(miss[:6]), ' ...' if len(miss) > 6 else ''))
def docs_of(out, arm, B):
    return sorted(int(re.search(r'doc(\d+)_', p).group(1)) for p in glob.glob('%s/%s/doc*_b%d.f32' % (out, arm, B)))
def pair(out, ref, test, B):
    """returns dict or None (only if either arm has no files at all for this bucket); rows matched by (doc,row)"""
    ds = sorted(set(docs_of(out, ref, B)) & set(docs_of(out, test, B)))
    kls, tops, dm = [], [], []
    for k in ds:
        a, b = load(out, ref, k, B), load(out, test, k, B)
        kl, st = kl_rows(a, b); kls.append(kl); tops.append(st); dm.append(kl.mean())
    if not kls: return None
    kl = np.concatenate(kls); st = np.concatenate(tops)
    return dict(docs=len(kls), rows=len(kl), dmean=float(kl.mean()), top_delta_pt=float(100 * (st.mean() - 1)), max=float(kl.max()),
                ci_row=ci(kl), ci_doc=ci(dm), kl=kl)
def crel(out, B):
    """BUNDLE3-style: KL(C||B)-KL(C||A) paired per row, ref = C"""
    ds = sorted(set(docs_of(out, 'A', B)) & set(docs_of(out, 'B', B)) & set(docs_of(out, 'C', B))); d = []; per = []
    for k in ds:
        c, a, b = load(out, 'C', k, B), load(out, 'A', k, B), load(out, 'B', k, B)
        ka, _ = kl_rows(c, a); kb, _ = kl_rows(c, b); d.append(kb - ka); per.append((kb - ka).mean())
    if not d: return None
    d = np.concatenate(d); return dict(docs=len(per), rows=len(d), dmean=float(d.mean()), ci_row=ci(d), ci_doc=ci(per))
def fmt(r, k=None):
    if r is None: return 'n/a'
    return 'rows=%d docs=%d dmean=%+.5f CIdoc=[%+.5f,%+.5f] CIrow=[%+.5f,%+.5f] top=%+.2fpt max=%.4f' % (r['rows'], r['docs'], r['dmean'], *r['ci_doc'], *r['ci_row'], r['top_delta_pt'], r['max']) if 'top_delta_pt' in r else \
        'rows=%d docs=%d dmean=%+.5f CIdoc=[%+.5f,%+.5f] CIrow=[%+.5f,%+.5f]' % (r['rows'], r['docs'], r['dmean'], *r['ci_doc'], *r['ci_row'])
def main(out):
    check_complete(out)
    md = ['# G1c result (analyzer output)', '']; fails = []; meaningless = []; ok = True; rowsok = True
    for B in BUCKETS:
        md.append('## bucket %d' % B)
        rb = pair(out, 'A', 'B', B); rp = pair(out, 'A', 'Ap', B); rc = pair(out, 'A', 'C', B); rbc = pair(out, 'B', 'C', B); cr = crel(out, B)
        md += ['- B vs A (gate): ' + fmt(rb), '- A vs Ap (noise): ' + fmt(rp), '- A vs C (F32 floor): ' + fmt(rc) + ('' if rc else ' (C did not reach this bucket or not run)'),
               '- B vs C: ' + fmt(rbc), '- C-relative dmean KL(C||B)-KL(C||A): ' + fmt(cr)]
        if rb is None: ok = False; fails.append((B, 'no data')); rowsok = False; md.append(''); continue
        if rb['rows'] < 500: rowsok = False; md.append('- WARNING: rows=%d below >=500' % rb['rows'])
        if rb['docs'] < 8: md.append('- NOTE: only %d docs (<8): doc-cluster CI is wide (t with %d df); rows within a doc are correlated' % (rb['docs'], rb['docs'] - 1))
        bad = rb['dmean'] > 0.0010 or rb['top_delta_pt'] < -0.3
        if bad:
            ok = False; fails.append((B, 'dmean %+.5f top %+.2fpt' % (rb['dmean'], rb['top_delta_pt'])))
            if rp and rp['dmean'] > 0 and 0.5 <= rb['dmean'] / max(rp['dmean'], 1e-12) <= 2.0 or (rp and rp['dmean'] >= rb['dmean']): meaningless.append(B)
        md.append('')
    if ok and rowsok: v = 'PASS (B vs A within gate in every bucket: dmean<=+.0010, same-top delta>=-0.3 pt)'
    elif ok: v = 'PASS on available data but sample below >=500 rows in some bucket (<500 rows): NOT a qualifying result'
    elif meaningless and len(meaningless) == len(fails): v = 'FAIL in buckets %s; A-vs-Ap noise is of similar size or larger: gate not meaningful at this depth (owner decision)' % [b for b, _ in fails]
    else: v = 'FAIL in buckets %s (%s); A-vs-Ap not of similar size in all failing buckets: outside gate' % ([b for b, _ in fails], '; '.join(x for _, x in fails))
    md.insert(2, '**Verdict: %s**' % v); md.insert(3, '')
    with open(os.path.join(out, 'G1C_TABLE.md'), 'w') as fh: fh.write('\n'.join(md) + '\n'); print('\n'.join(md))
def selftest(out_fake, g1):
    """CPU sanity: analyzer KL vs G1's own cmp on real G1 .npy rows, then synthetic gate paths"""
    fs = sorted(glob.glob(g1 + '/A/*_r0_*.npy'))[:3]; n = 0
    for f in fs:
        a = np.load(f); b = np.load(f.replace('/A/', '/B/')) if os.path.exists(f.replace('/A/', '/B/')) else None
        if b is None: continue
        kl, st = kl_rows(a[None], b[None]); n += 1; print('selftest G1', os.path.basename(f), 'KL(A||B)=%.5f same_top=%s' % (kl[0], bool(st[0])))
    rng = np.random.default_rng(0); nv = NV
    for arm, eps in (('A', 0), ('B', 0.01), ('Ap', 0.3), ('C', 0.05)):
        os.makedirs('%s/%s' % (out_fake, arm), exist_ok=True)
        for k in range(8):
            for B in BUCKETS:
                base = np.random.default_rng(k * 7 + B).normal(0, 3, (NR, nv)).astype(np.float32)
                (base + eps * rng.normal(0, 1, base.shape).astype(np.float32)).tofile('%s/%s/doc%d_b%d.f32' % (out_fake, arm, k, B))
    main(out_fake)
if __name__ == '__main__':
    try:
        if '--selftest' in sys.argv: selftest(sys.argv[1], sys.argv[sys.argv.index('--selftest') + 1])
        else: main(sys.argv[1])
    except AnalyzerError as e:
        print('ANALYZER ERROR: %s' % e, file=sys.stderr); sys.exit(2)
