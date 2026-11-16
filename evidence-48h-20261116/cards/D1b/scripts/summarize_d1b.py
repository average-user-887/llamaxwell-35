#!/usr/bin/env python3
"""D1b summarizer. usage: summarize_d1b.py <run dir>. Arms prod (graphs env absent) vs gr (GGML_CUDA_MAXWELL_GRAPHS=1), same counter lib.
Per session: makespan, peak VRAM, per-slot token identity vs first prod session, d1b_graphs counter (last line in server.log). Accept lines at end."""
import sys, json, glob, re, os, statistics as st
sys.path.insert(0, '<W>/D0')
import compare as C
run = sys.argv[1]; vr = []
try:
    for l in open(run + '/vram.csv'):
        p = l.strip().split(',')
        try: vr.append((float(p[0]), float(p[1])))
        except Exception: pass
except OSError: pass
CRE = re.compile(r'd1b_graphs\[(?:run|exit)\]: compute_calls=(\d+) arch_off=(\d+) captures=(\d+) graph_launches=(\d+)')
S = []
for f in sorted(glob.glob(run + '/runs/b*/run.json')):
    R = json.load(open(f))
    if R.get('kind') != 'main' or not R.get('ok') or not R.get('reqs'): print('skipped (not ok):', f, R.get('error'), file=sys.stderr); continue
    rq = R['reqs']; mk = max(r['t_end'] for r in rq) - R['t_run0']; d = os.path.dirname(f)
    m = CRE.findall(open(d + '/server.log', errors='replace').read()); c = tuple(int(x) for x in m[-1]) if m else (0, 0, 0, 0)
    t0, t1 = R.get('utc_start', R['t_run0']), R.get('utc_end', max(r['t_end'] for r in rq))
    pk = max([v for t, v in vr if t0 <= t <= t1], default=float('nan'))
    S.append(dict(tag=R['tag'], arm=R['arm'], mk=mk, peak=pk, ids=C.load(f), cnt=c, nl=len(m)))
S.sort(key=lambda s: s['tag']); ref = next((s for s in S if s['arm'] == 'prod'), None)
def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else float('nan')) if v else (float('nan'),) * 2
print('| arm | n | makespan mean s | sd s | peak VRAM max MiB | token-identical to prod ref | compute_calls (mean) | graph_launches (mean) | captures (mean) |\n|---|---|---|---|---|---|---|---|---|')
for a in ('prod', 'gr'):
    X = [s for s in S if s['arm'] == a]
    if not X: continue
    idt = sum(1 for s in X if ref and C.first_div(ref['ids'], s['ids']) is None)
    print('| %s | %d | %.2f | %.2f | %.0f | %d/%d | %.0f | %.0f | %.0f |' % (a, len(X), *ms([s['mk'] for s in X]), max(s['peak'] for s in X), idt, len(X),
          *[st.mean(s['cnt'][i] for s in X) for i in (0, 3, 2)]))
P = [s for s in S if s['arm'] == 'prod']; G = [s for s in S if s['arm'] == 'gr']
if P and G:
    mp, sp = ms([s['mk'] for s in P]); mg, sg = ms([s['mk'] for s in G]); gain = (mp - mg) / mp * 100; diff = mp - mg
    z1 = diff / sp if sp == sp and sp > 0 else float('nan'); se = ((sp ** 2 + sg ** 2) / min(len(P), len(G))) ** 0.5 if sp == sp and sg == sg else float('nan'); z2 = diff / se if se else float('nan')
    pp = max(s['peak'] for s in P); pg = max(s['peak'] for s in G)
    print('\nN3 makespan gain gr vs prod: %+.2f%% (%.2f s); diff/sd_prod = %.1f, Welch z = %.1f' % (gain, diff, z1, z2))
    print('ACCEPT gain >= 1%% and >= 3 sd: %s' % ('PASS' if gain >= 1 and z1 >= 3 else 'FAIL'))
    print('ACCEPT token identity gr == prod ref (all slots/turns): %s' % ('PASS' if all(ref and C.first_div(ref['ids'], s['ids']) is None for s in G) else 'FAIL'))
    print('ACCEPT peak VRAM gr %.0f <= prod %.0f + 100 MiB: %s' % (pg, pp, 'PASS' if pg <= pp + 100 else 'FAIL'))
    print('ACCEPT engagement: gr graph_launches > 0 (min %d), prod graph_launches == 0 (max %d): %s' % (min(s['cnt'][3] for s in G), max(s['cnt'][3] for s in P),
          'PASS' if all(s['cnt'][3] > 0 for s in G) and all(s['cnt'][3] == 0 for s in P) else 'FAIL (void arm)'))
print('\nPer session:\n\n| session | arm | makespan s | peak MiB | calls | arch_off | captures | launches | per-slot identity vs prod ref |\n|---|---|---|---|---|---|---|---|---|')
for s in S:
    sm = C.slot_summary(ref['ids'], s['ids']) if ref else {}
    print('| %s | %s | %.2f | %.0f | %d | %d | %d | %d | %s |' % (s['tag'], s['arm'], s['mk'], s['peak'], *s['cnt'], '; '.join('slot%s %s' % (k, v) for k, v in sm.items())))
print('\nmakespan = last request end - t_run0; peak = max whole-GPU 5 s sample in the session window. Counters from the last d1b_graphs line of each server.log (periodic every 5000 launches + exit).')
