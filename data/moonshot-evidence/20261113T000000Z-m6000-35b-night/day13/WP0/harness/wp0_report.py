#!/usr/bin/env python3
"""WP0 report: singles (+4K at 90K), paired makespan full/gemm vs b3, VRAM peaks + headroom, GEMM counters per run, Q8 slot round trip. usage: wp0_report.py <outdir> <RESULT.md>"""
import sys, os, json, glob, re, statistics as st, importlib.util
out, res = sys.argv[1], sys.argv[2]; H = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location('an', H + '/n3_analyze.py'); an = importlib.util.module_from_spec(sp); sp.loader.exec_module(an)
runs = an.load(out); main = [r for r in runs if r.get('kind') == 'main']; sing = [r for r in runs if r.get('kind') == 'single']
GEMM_RE = re.compile(r'occamy_fa_gemm: engaged=(\d+) fallback=(\d+) reasons=(\S+)')
VRAM_TOTAL = 24576
L = ['# WP0 result: candidate = WP0 build (2436f75 + cap patch + GEMM v4 + counters + KVQ8)', '',
     'Arms: b3 = G1b binary + bundle3 lib, F16 KV, ub1024, cap0 (prod-equivalent). gemm = WP0 build, GEMM on, F16 KV, ub1024, cap0. full = WP0 build, GEMM on, K q8_0 / V f16 (KVQ8), ub2048, cap1024 (adoption candidate). Short profile.', '']
L.append('Runs ok: ' + ', '.join('%s=%s' % (r['tag'], r.get('ok')) for r in runs))
errs = [(r['tag'], r.get('error')) for r in runs if r.get('error')]
if errs: L.append('ERRORS: %s' % errs)
L += ['', '## Single slot: +4K append at 90K (restore 84K, append to 90K, then 90K->94K)', '', '| run | n_restored | 84K->90K prompt_ms | 90K->94K prompt_ms | wall s | engaged | fallback | reasons | peak VRAM MiB |', '|---|---|---|---|---|---|---|---|---|']
for r in sing:
    s = r.get('steps', []); g = r.get('gemm_check') or {}
    lg = '%s/runs/%s/server.log' % (out, r['tag']); m = GEMM_RE.findall(open(lg, errors='replace').read()) if os.path.exists(lg) else []
    e, fb, rs = m[-1] if m else ('-', '-', '-')
    L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (r['tag'], r.get('n_restored'), an.f(s[0]['prompt_ms'], 0) if len(s) > 0 else '-', an.f(s[1]['prompt_ms'], 0) if len(s) > 1 else '-', an.f(s[1]['wall_s'], 2) if len(s) > 1 else '-', e, fb, rs, an.f(an.telem('%s/runs/%s/telemetry.csv' % (out, r['tag']), 0, 9e12).get('vram_peak_mib'), 0)))
L.append('\nAccept (WP0): gemm single +4K at 90K within 23.4 s +-3% (22.70..24.10 s) and engaged > 0.')
L += ['', '## Two-slot N3 makespan (s), paired per block', '', '| block | b3 | gemm | full | prod | gemm vs b3 | full vs b3 | prod vs b3 |', '|---|---|---|---|---|---|---|---|']
dd = {'gemm': [], 'full': [], 'prod': []}
for blk in sorted({r['block'] for r in main}):
    v = {a: next((r['M'].get('makespan_total_s') for r in main if r['block'] == blk and r['arm'] == a), None) for a in ('b3', 'gemm', 'full', 'prod')}
    row = []
    for a in ('gemm', 'full', 'prod'):
        if v['b3'] and v[a]: dd[a].append(100 * (v[a] / v['b3'] - 1)); row.append('%+.2f%%' % dd[a][-1])
        else: row.append('-')
    L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (blk, an.f(v['b3'], 2), an.f(v['gemm'], 2), an.f(v['full'], 2), an.f(v['prod'], 2), row[0], row[1], row[2]))
for a in ('gemm', 'full', 'prod'):
    if len(dd[a]) > 1: m, lo, hi = an.ci(dd[a]); L.append('- %s vs b3: paired mean %+.2f%% (CI95 %+.2f..%+.2f, n=%d)' % (a, m, lo, hi, len(dd[a])))
    elif dd[a]: L.append('- %s vs b3: %+.2f%% (n=1)' % (a, dd[a][0]))
L.append('- reference (N3B, 3 blocks, same short profile): b3 210.72 s, whole-v4 lib 177.67 s (-15.69%)')
L += ['', '## Per arm means (main)', '', '| arm | makespan s | prefill s (server) | decode s | max gap s | peak VRAM MiB | headroom MiB (of %d) |' % VRAM_TOTAL, '|---|---|---|---|---|---|---|']
for a in ('b3', 'gemm', 'full', 'prod'):
    x = [r['M'] for r in main if r['arm'] == a and r['M']]
    if x:
        pk = max(m.get('vram_peak_mib', 0) for m in x)
        L.append('| %s | %.2f | %.2f | %.2f | %s | %d | %d |' % (a, st.mean(m['makespan_total_s'] for m in x), st.mean(m['prefill_s'] for m in x), st.mean(m['decode_s'] for m in x), an.f(st.mean(m.get('max_gap_s') or 0 for m in x), 2), pk, VRAM_TOTAL - pk))
L += ['', '## GEMM counters per run (last occamy_fa_gemm line, printed at backend free)', '', '| run | engaged | fallback | reasons |', '|---|---|---|---|']
for r in runs:
    lg = '%s/runs/%s/server.log' % (out, r['tag'])
    if r['arm'] in ('gemm', 'full', 'prod') and os.path.exists(lg):
        m = GEMM_RE.findall(open(lg, errors='replace').read()); e, fb, rs = m[-1] if m else ('MISSING', '-', '-'); L.append('| %s | %s | %s | %s |' % (r['tag'], e, fb, rs))
L += ['', '## Q8 slot set (full arm): restore count + round trip (16 tokens decoded twice after erase+restore)', '']
for f in glob.glob(H + '/slots-q8/n3_slots_*.json'):
    j = json.load(open(f)); L.append('- %s: n_restored=%s, build_s=%.0f, roundtrip=%s' % (os.path.basename(f), [s.get('n_restored') for s in j['slots']], j.get('build_s', 0), j.get('roundtrip')))
L += ['', '## Receipts', '', '- %s' % '; '.join('%s: %d/%d receipt_ok' % (r['tag'], sum(1 for q in r.get('reqs', []) if q.get('receipt_ok')), len(r.get('reqs', []))) for r in main)]
open(res, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
