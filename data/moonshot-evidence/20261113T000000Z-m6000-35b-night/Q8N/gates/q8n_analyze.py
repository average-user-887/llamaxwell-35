#!/usr/bin/env python3
"""Q8N exactness analyzer. usage: python3 -I q8n_analyze.py <OUT>
Rows <OUT>/<ARM>/doc<k>_b<B>.f32 (float32 [NR, n_vocab]). KL / CI code imported unchanged from G1c g1c_analyze.py.
Gate (N vs G, same q8 cache content, same schedule): dmean <= +.0010 and same-top delta >= -0.3 pt in EVERY bucket, read against Gp-vs-G noise.
Also: max |logit diff| N vs G, X-arm per-call native-vs-generic diff (occamy_fa_cmp_kvq8 lines), q8-vs-F16 context (G vs F, N vs F), mixed arms vs F."""
import sys, os, re, glob, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('g1c', '<LAB>/occamy35-day-20261112/evidence/G1c/g1c_analyze.py')
g1c = importlib.util.module_from_spec(spec); spec.loader.exec_module(g1c)
BUCKETS = tuple(int(x) for x in os.environ.get('Q8N_BUCKETS', '4096,32768,98304').split(',')); NR = int(os.environ.get('Q8N_NR', 64))
def docs_of(out, arm, B): return sorted(int(re.search(r'doc(\d+)_', p).group(1)) for p in glob.glob('%s/%s/doc*_b%d.f32' % (out, arm, B)))
def pair(out, ref, test, B):
    kls, tops, dm, mx = [], [], [], 0.0
    for k in sorted(set(docs_of(out, ref, B)) & set(docs_of(out, test, B))):
        a = np.fromfile('%s/%s/doc%d_b%d.f32' % (out, ref, k, B), np.float32); b = np.fromfile('%s/%s/doc%d_b%d.f32' % (out, test, k, B), np.float32)
        if a.size == 0 or a.size != b.size or a.size % NR: continue
        a = a.reshape(NR, -1); b = b.reshape(NR, -1); kl, st = g1c.kl_rows(a, b)
        kls.append(kl); tops.append(st); dm.append(kl.mean()); mx = max(mx, float(np.abs(a - b).max()))
    if not kls: return None
    kl = np.concatenate(kls); st = np.concatenate(tops)
    return dict(docs=len(kls), rows=len(kl), dmean=float(kl.mean()), top=float(100 * (st.mean() - 1)), klmax=float(kl.max()), ci_doc=g1c.ci(dm), ci_row=g1c.ci(kl), maxabs=mx)
def fmt(r):
    if r is None: return 'n/a'
    return 'rows=%d docs=%d dmean=%+.6f CIdoc=[%+.6f,%+.6f] top=%+.2fpt klmax=%.5f max|dlogit|=%.4g' % (r['rows'], r['docs'], r['dmean'], *r['ci_doc'], r['top'], r['klmax'], r['maxabs'])
def xlog(out):
    rx = re.compile(r'occamy_fa_cmp_kvq8 call=(\d+) n_kv=(\d+) ns=(\d+) max_abs=(\S+) max_ref=(\S+) max_rel=(\S+) nan=(\d+)')
    rows = [m.groups() for f in glob.glob(out + '/X/log_*.txt') for m in rx.finditer(open(f, errors='replace').read())]
    if not rows: return 'X: no occamy_fa_cmp_kvq8 lines (arm not run or path inactive)'
    ma = max(float(r[3]) for r in rows); mr = max(float(r[5]) for r in rows); nan = sum(int(r[6]) for r in rows); nk = max(int(r[1]) for r in rows)
    return 'X (native vs generic, same cache, per FA call): calls_logged=%d max n_kv=%d max_abs=%.3e max_rel=%.3e nan=%d' % (len(rows), nk, ma, mr, nan)
def main(out):
    md = ['# Q8N exactness (analyzer output)', '']; ok = True; fails = []
    for B in BUCKETS:
        g = pair(out, 'G', 'N', B); n = pair(out, 'G', 'Gp', B)
        md += ['## bucket %d' % B, '- N vs G (gate): ' + fmt(g), '- Gp vs G (noise): ' + fmt(n), '- G vs F (q8 cache cost, generic): ' + fmt(pair(out, 'F', 'G', B)),
               '- N vs F (q8 cache cost, native): ' + fmt(pair(out, 'F', 'N', B))]
        for a in ('Nk', 'Nv'):
            r = pair(out, 'F', a, B)
            if r: md.append('- %s vs F (mixed, info): %s' % (a, fmt(r)))
        if g is None: ok = False; fails.append('%d no data' % B)
        elif g['dmean'] > 0.0010 or g['top'] < -0.3: ok = False; fails.append('%d dmean %+.6f top %+.2fpt' % (B, g['dmean'], g['top']))
        elif g['rows'] < 256: md.append('- NOTE: rows=%d' % g['rows'])
        md.append('')
    md.append('- ' + xlog(out))
    md.insert(2, '**Verdict: %s**' % ('PASS (N vs G within dmean<=+.0010, top>=-0.3 pt in every bucket)' if ok else 'FAIL: ' + '; '.join(fails))); md.insert(3, '')
    open(os.path.join(out, 'Q8N_EXACT.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
if __name__ == '__main__': main(sys.argv[1])
