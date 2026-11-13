#!/usr/bin/env python3
"""N3B report. usage: n3b_report.py <outdir> <RESULT.md>. Per-request prompt_ms by arm/kind/slot, paired makespan, VRAM, nvprof kernel engagement, server.log gemm lines."""
import sys, os, json, glob, csv, re, statistics as st, importlib.util
out, res = sys.argv[1], sys.argv[2]
sp = importlib.util.spec_from_file_location('an', os.path.dirname(os.path.abspath(__file__)) + '/n3_analyze.py'); an = importlib.util.module_from_spec(sp); sp.loader.exec_module(an)
runs = an.load(out); main = [r for r in runs if r['kind'] == 'main']; L = ['# N3B result (gemm = integration-v4 lib sha 0c7ffbde, b3 = bundle3; short profile, ub 1024, cap 0, 3 blocks)', '']
L.append('Runs ok: ' + ', '.join('%s=%s' % (r['tag'], r.get('ok')) for r in runs)); errs = [(r['tag'], r.get('error')) for r in runs if r.get('error')]
if errs: L.append('ERRORS: %s' % errs)
L += ['', '## prompt_ms per request (agent=slot, turn: kind, prompt_n) by block', '', '| block | arm | slot | turn | kind | prompt_n | prompt_ms | tok/s |', '|---|---|---|---|---|---|---|---|']
agg = {}
for r in sorted(main, key=lambda r: (r['block'], r['tag'])):
    for q in sorted(r.get('reqs', []), key=lambda q: (q['agent'], q['turn'])):
        pm = q.get('prompt_ms'); L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (r['block'], r['arm'], q['agent'], q['turn'], q['kind'], q.get('prompt_n'), an.f(pm, 1), an.f(q.get('prompt_tps'), 1)))
        if pm is not None: agg.setdefault((r['arm'], q['kind'], q['agent']), []).append(pm); agg.setdefault((r['arm'], q['kind'], 'both'), []).append(pm)
L += ['', '## mean prompt_ms by arm / size / slot (n)', '', '| kind | slot | b3 | gemm | gemm vs b3 |', '|---|---|---|---|---|']
for k in ('1K', '4K'):
    for s in (0, 1, 'both'):
        a, b = agg.get(('b3', k, s)), agg.get(('gemm', k, s))
        if a and b: L.append('| %s | %s | %.1f (%d) | %.1f (%d) | %+.2f%% |' % (k, s, st.mean(a), len(a), st.mean(b), len(b), 100 * (st.mean(b) / st.mean(a) - 1)))
L += ['', '## makespan (total s) per block, paired', '', '| block | b3 | gemm | gemm-b3 s | % |', '|---|---|---|---|---|']
d = []
for blk in sorted({r['block'] for r in main}):
    a = [r['M'].get('makespan_total_s') for r in main if r['block'] == blk and r['arm'] == 'b3']; b = [r['M'].get('makespan_total_s') for r in main if r['block'] == blk and r['arm'] == 'gemm']
    if a and b and a[0] and b[0]: d.append(100 * (b[0] / a[0] - 1)); L.append('| %s | %.2f | %.2f | %+.2f | %+.2f%% |' % (blk, a[0], b[0], b[0] - a[0], d[-1]))
if len(d) > 1: m, lo, hi = an.ci(d); L.append('\nPaired mean %+.2f%% (CI95 %+.2f..%+.2f, n=%d)' % (m, lo, hi, len(d)))
elif d: L.append('\nPaired %+.2f%% (n=1)')
for arm in ('b3', 'gemm'):
    x = [r['M'] for r in main if r['arm'] == arm and r['M']]
    if x: L.append('- %s: makespan mean %.2f s; prefill_s(server) mean %.2f; peak VRAM %s MiB' % (arm, st.mean(m['makespan_total_s'] for m in x), st.mean(m['prefill_s'] for m in x), max(m.get('vram_peak_mib', 0) for m in x)))
L += ['', '## Engagement (nvprof, gemm arm, prof_gemm session)', '']
tr = out + '/prof/trace.csv'
try:
    rows = list(csv.reader(open(tr, errors='replace'))); h = next(i for i, r in enumerate(rows) if r and r[0] == 'Start'); hd = rows[h]; ui = hd.index('Duration'); ni = hd.index('Name')
    f = {'s': 1e3, 'ms': 1.0, 'us': 1e-3, 'ns': 1e-6}[rows[h + 1][ui]]; cnt = {}
    for r in rows[h + 2:]:
        if len(r) > ni:
            try: t = float(r[ui]) * f
            except ValueError: continue
            n = re.sub(r'\(.*', '', r[ni])[:110]; c = cnt.setdefault(n, [0, 0.0]); c[0] += 1; c[1] += t
    tot = sum(v[1] for v in cnt.values())
    def grp(p): return [(k, v) for k, v in cnt.items() if re.search(p, k)]
    for lab, p in (('sgemm', r'sgemm'), ('masked_softmax', r'softmax'), ('flash_attn_tile', r'flash_attn_tile'), ('flash_attn (any)', r'flash_attn')):
        g = grp(p); L.append('- %s: %d launches, %.1f ms (%.1f%% of %.0f ms GPU)' % (lab, sum(v[0] for _, v in g), sum(v[1] for _, v in g), 100 * sum(v[1] for _, v in g) / tot if tot else 0, tot))
    L += ['', 'Top 12 kernels by time:', ''] + ['- %s: n=%d, %.1f ms' % (k, v[0], v[1]) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1][1])[:12]]
except Exception as ex: L.append('trace unavailable: %r' % ex)
for tag in ('prof_gemm',) + tuple(r['tag'] for r in main if r['arm'] == 'gemm')[:1]:
    lg = '%s/runs/%s/server.log' % (out, tag)
    if os.path.exists(lg):
        g = [l.strip()[:220] for l in open(lg, errors='replace') if re.search(r'gemm|GEMM|fallback', l)][:12]; L += ['', 'server.log (%s) gemm/fallback lines (%d shown):' % (tag, len(g))] + ['    ' + x for x in g]
L += ['', 'Verdict: see engagement counts above; raw N3 analyzer table in N3_TABLE.md.']
open(res, 'w').write('\n'.join(L) + '\n')
