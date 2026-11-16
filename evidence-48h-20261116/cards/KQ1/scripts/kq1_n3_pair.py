#!/usr/bin/env python3
"""KQ1 N3 paired summary. usage: kq1_n3_pair.py <run dir>   (sessions <run>/prod_<p>/ and <run>/kq1_<p>/, one main N3 short session each)
Metrics from WP0 n3_analyze.metrics (unchanged). Gates (fixed, card KQ1): makespan gain >= 1% and diff/sd_prod >= 3 (D1b convention; Welch z also shown);
prefill_s (server) kq1 <= prod +0.2%. Engagement: occamy_kq1 stage>0 and occamy_fa_gemm kq8>0 in every kq1 server.log, else arm void.
Token identity: per-slot first divergence vs the first prod session (D0 compare.py), reported, not gated."""
import sys, os, re, glob, json, statistics as st, importlib.util
H = '<M6000_ROOT>/occamy35-night-20261113/WP0/harness/n3_analyze.py'
spec = importlib.util.spec_from_file_location('n3a', H); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
sys.path.insert(0, '<W>/D0'); import compare as C
run = sys.argv[1]
KQ = re.compile(r'occamy_kq1: stage=(\d+) kvq8_plain=(\d+) stage_misalign=(\d+) stock_kq8=(\d+)'); GQ = re.compile(r'occamy_fa_gemm: engaged=(\d+) q_tiled=(\d+) kq8=(\d+) fallback=(\d+)')
MET = [('makespan_total_s', 'makespan total s', -1), ('ttft_mean_s', 'TTFT mean s', -1), ('prefill_s', 'prefill s (server)', -1), ('decode_s', 'decode s (server)', -1),
       ('cw_decode_tps', 'common-window decode tok/s', +1), ('max_gap_s', 'max gap s', -1), ('vram_peak_mib', 'peak VRAM MiB', -1)]
S = {'prod': [], 'kq1': []}
for d in sorted(glob.glob(run + '/*_[0-9]*')):
    arm = os.path.basename(d).rsplit('_', 1)[0]
    if arm not in S or not os.path.isdir(d): continue
    for R in A.load(d):
        if R['kind'] != 'main': continue
        f = glob.glob('%s/runs/%s/server.log' % (d, R['tag'])); log = open(f[0], errors='replace').read() if f else ''
        k = [tuple(map(int, m)) for m in KQ.findall(log)]; g = [tuple(map(int, m)) for m in GQ.findall(log)]
        S[arm].append(dict(dir=d, ok=bool(R.get('ok')), M=R['M'], kq1=k[-1] if k else None, gemm=g[-1] if g else None, ids=C.load(d)))
def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else float('nan')) if v else (float('nan'),) * 2
P, K = [s for s in S['prod'] if s['ok']], [s for s in S['kq1'] if s['ok']]
print('# KQ1 N3 paired summary\n\nsessions ok: prod %d/%d, kq1 %d/%d\n' % (len(P), len(S['prod']), len(K), len(S['kq1'])))
print('| metric | prod mean | prod sd | kq1 mean | kq1 sd | kq1 vs prod % |\n|---|---|---|---|---|---|')
res = {}
for k, lab, sign in MET:
    a = [s['M'][k] for s in P if s['M'].get(k) is not None]; b = [s['M'][k] for s in K if s['M'].get(k) is not None]
    (ma, sa), (mb, sb) = ms(a), ms(b); pct = 100 * (mb - ma) / ma if a and b and ma else float('nan'); res[k] = (ma, sa, mb, sb, pct, len(a), len(b))
    print('| %s | %.3f | %.3f | %.3f | %.3f | %+.2f |' % (lab, ma, sa, mb, sb, pct))
ma, sa, mb, sb, pct, na, nb = res['makespan_total_s']; diff = ma - mb; gain = 100 * diff / ma if ma == ma and ma else float('nan')
z1 = diff / sa if sa == sa and sa > 0 else float('nan'); se = ((sa ** 2) / max(na, 1) + (sb ** 2) / max(nb, 1)) ** 0.5 if sa == sa and sb == sb else float('nan'); z2 = diff / se if se == se and se > 0 else float('nan')
eng = bool(K) and all(s['kq1'] and s['kq1'][0] > 0 and s['gemm'] and s['gemm'][2] > 0 for s in K)
pf = res['prefill_s']; pf_ok = pf[4] == pf[4] and pf[4] <= 0.2
print('\n## counters (last line per kq1 session: occamy_kq1 stage,kvq8_plain,stage_misalign,stock_kq8 | occamy_fa_gemm engaged,q_tiled,kq8,fallback)')
for s in S['kq1']: print('- %s ok=%s kq1=%s gemm=%s' % (os.path.basename(s['dir']), s['ok'], s['kq1'], s['gemm']))
for s in S['prod']: print('- %s ok=%s kq1=%s gemm=%s (prod: kq1 stage must be 0)' % (os.path.basename(s['dir']), s['ok'], s['kq1'], s['gemm']))
print('\n## token identity vs first prod session (per slot)')
ref = P[0]['ids'] if P else {}
for s in P[1:] + K:
    print('- %s: %s' % (os.path.basename(s['dir']), '; '.join('slot %d %s' % (a, v) for a, v in C.slot_summary(ref, s['ids']).items()) if ref else 'no ref'))
print('\n## gates')
print('- engagement (stage>0 and gemm kq8>0 in every kq1 session): %s' % ('PASS' if eng else 'FAIL -> arm VOID'))
print('- N3 makespan gain kq1 vs prod: %+.2f%% (%.2f s); diff/sd_prod = %.1f, Welch z = %.1f; n prod %d kq1 %d -> %s' % (gain, diff, z1, z2, na, nb, 'PASS' if gain >= 1 and z1 >= 3 else 'FAIL'))
print('- prefill s kq1 vs prod: %+.2f%% (limit +0.2%%) -> %s' % (pf[4], 'PASS' if pf_ok else 'FAIL'))
json.dump({k: v for k, v in res.items()} | {'gain_pct': gain, 'z1': z1, 'z2': z2, 'engaged': eng, 'prefill_ok': pf_ok}, open(run + '/kq1_n3_summary.json', 'w'), indent=1)
