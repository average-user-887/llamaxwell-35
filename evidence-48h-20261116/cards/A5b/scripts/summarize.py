#!/usr/bin/env python3
"""A5b summarizer. usage: summarize.py <run dir>. Table first: makespan mean/sd per policy (LLAMA_N3_SCHED 0/3/4), peak VRAM and worst visible token gap per session with the
accept lines (peak <= p0 peak + 100 MiB; worst visible gap <= 10 s), token identity vs the first p0 session, engagement (n3_sched / n3_sched_hold lines from server.log)."""
import sys, json, glob, re, os, csv, statistics as st
sys.path.insert(0, '<W>/D0')
import compare as C
run = sys.argv[1]; S = []
vr = []
try:
    for l in open(run + '/vram.csv'):
        p = l.strip().split(',')
        try: vr.append((float(p[0]), float(p[1])))
        except Exception: pass
except OSError: pass
def gaps(rq):
    g = 0.0
    for r in rq:
        t = [e[0] for e in (r.get('ev') or [])]
        for a, b in zip(t, t[1:]): g = max(g, b - a)
    return g
for f in sorted(glob.glob(run + '/runs/b*/run.json')):
    R = json.load(open(f))
    if R.get('kind') != 'main' or not R.get('ok') or not R.get('reqs'): print('skipped (not ok):', f, R.get('error'), file=sys.stderr); continue
    rq = R['reqs']; mk = max(r['t_end'] for r in rq) - R['t_run0']
    sl = {a: max(r['t_end'] for r in rq if r['agent'] == a) - R['t_run0'] for a in (0, 1)}
    pol = int(R['arm'][1:]); d = os.path.dirname(f); log = open(d + '/server.log', errors='replace').read()
    eng = ' '.join(re.findall(r'n3_sched(?:_hold)?: .*', log)[-2:]) if pol else '(none; policy 0)'
    if pol and 'n3_sched' not in eng: eng = 'MISSING'
    t0, t1 = R.get('utc_start', R['t_run0']), R.get('utc_end', max(r['t_end'] for r in rq))
    pk = max([m for t, m in vr if t0 <= t <= t1], default=float('nan'))
    S.append(dict(f=f, tag=R['tag'], pol=pol, mk=mk, sl=sl, gap=gaps(rq), peak=pk, ids=C.load(f), eng=eng))
S.sort(key=lambda s: s['tag'])
ref = next((s for s in S if s['pol'] == 0), None)
def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else float('nan'))
p0pk = max([s['peak'] for s in S if s['pol'] == 0 and s['peak'] == s['peak']], default=float('nan'))
m0 = ms([s['mk'] for s in S if s['pol'] == 0])[0] if ref else float('nan')
print('| policy | n | makespan mean s | sd s | vs policy 0 | slot0 end mean s | slot1 end mean s | peak VRAM max MiB | worst gap max s | token-identical to p0 ref |\n|---|---|---|---|---|---|---|---|---|---|')
for p in range(5):
    X = [s for s in S if s['pol'] == p]
    if not X: continue
    mean, sd = ms([s['mk'] for s in X]); idt = sum(1 for s in X if C.first_div(ref['ids'], s['ids']) is None) if ref else 0
    print('| %d | %d | %.2f | %.2f | %+.2f%% | %.2f | %.2f | %.0f | %.2f | %d/%d |' % (p, len(X), mean, sd, 100 * (mean / m0 - 1), st.mean(s['sl'][0] for s in X), st.mean(s['sl'][1] for s in X),
          max(s['peak'] for s in X), max(s['gap'] for s in X), idt, len(X)))
print()
for p in (4, 3):
    X = [s for s in S if s['pol'] == p]
    if not X: continue
    pk = max(s['peak'] for s in X); gp = max(s['gap'] for s in X)
    print('ACCEPT policy %d peak VRAM %.0f <= p0 peak %.0f + 100 MiB: %s' % (p, pk, p0pk, 'PASS' if pk <= p0pk + 100 else ('FAIL' if pk == pk else 'NO DATA')))
    print('ACCEPT policy %d worst visible gap %.2f s <= 10 s: %s' % (p, gp, 'PASS' if gp <= 10 else 'FAIL'))
print('(p0 worst gap max %.2f s)' % max([s['gap'] for s in S if s['pol'] == 0], default=float('nan')))
print('\nPer session (tag b<block>_p<position>):\n\n| session | policy | makespan s | slot0 end s | slot1 end s | peak MiB | worst gap s | identity vs p0 ref | engagement |\n|---|---|---|---|---|---|---|---|---|')
for s in S:
    dv = C.first_div(ref['ids'], s['ids']) if ref else None; sm = C.slot_summary(ref['ids'], s['ids']) if ref else {}
    ident = 'identical' if dv is None else 'slot0: %s; slot1: %s' % (sm.get(0), sm.get(1))
    print('| %s | %d | %.2f | %.2f | %.2f | %.0f | %.2f | %s | %s |' % (s['tag'], s['pol'], s['mk'], s['sl'][0], s['sl'][1], s['peak'], s['gap'], ident, s['eng'].replace('n3_sched: ', '').replace('n3_sched_hold: ', 'HOLD ')))
print('\nReference = %s. makespan = last request end - t_run0. Peak = max of 5 s whole-GPU samples inside the session window. gap = max inter-event gap within one streamed request.' % (ref['tag'] if ref else 'n/a'))
