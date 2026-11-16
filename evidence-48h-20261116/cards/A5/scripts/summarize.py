#!/usr/bin/env python3
"""A5 summarizer. usage: summarize.py <run dir>. Table first: makespan (s, from first request start t_run0 to last request end) mean/sd per policy (LLAMA_N3_SCHED),
engagement counters from server.log (n3_sched line), per-slot token identity vs policy 0 (ref = first policy-0 session; also p0-vs-p0 determinism)."""
import sys, json, glob, re, os, statistics as st
sys.path.insert(0, '<W>/D0')
import compare as C
run = sys.argv[1]; S = []
for f in sorted(glob.glob(run + '/runs/b*/run.json')):
    R = json.load(open(f))
    if R.get('kind') != 'main' or not R.get('ok') or not R.get('reqs'): print('skipped (not ok):', f, R.get('error'), file=sys.stderr); continue
    rq = R['reqs']; mk = max(r['t_end'] for r in rq) - R['t_run0']
    sl = {a: max(r['t_end'] for r in rq if r['agent'] == a) - R['t_run0'] for a in (0, 1)}
    pol = int(R['arm'][1:]); log = open(os.path.dirname(f) + '/server.log', errors='replace').read()
    m = re.findall(r'n3_sched: .*', log)
    S.append(dict(f=f, tag=R['tag'], pol=pol, mk=mk, sl=sl, ids=C.load(f), eng=m[-1] if m else ('(none; policy 0)' if pol == 0 else 'MISSING')))
S.sort(key=lambda s: s['tag'])
ref = next((s for s in S if s['pol'] == 0), None)
def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else float('nan'))
print('| policy | n | makespan mean s | sd s | vs policy 0 | slot0 end mean s | slot1 end mean s | token-identical to p0 ref |\n|---|---|---|---|---|---|---|---|')
m0 = ms([s['mk'] for s in S if s['pol'] == 0])[0] if ref else float('nan')
for p in range(4):
    X = [s for s in S if s['pol'] == p]
    if not X: print('| %d | 0 | - | - | - | - | - | - |' % p); continue
    mean, sd = ms([s['mk'] for s in X]); idt = sum(1 for s in X if C.first_div(ref['ids'], s['ids']) is None) if ref else 0
    print('| %d | %d | %.2f | %.2f | %+.2f%% | %.2f | %.2f | %d/%d |' % (p, len(X), mean, sd, 100 * (mean / m0 - 1), st.mean(s['sl'][0] for s in X), st.mean(s['sl'][1] for s in X), idt, len(X)))
print('\nPer session (block order = tag b<block>_p<position>):\n\n| session | policy | makespan s | slot0 end s | slot1 end s | identity vs p0 ref | engagement |\n|---|---|---|---|---|---|---|')
for s in S:
    d = C.first_div(ref['ids'], s['ids']) if ref else None; sm = C.slot_summary(ref['ids'], s['ids']) if ref else {}
    ident = 'identical' if d is None else 'slot0: %s; slot1: %s' % (sm.get(0), sm.get(1))
    print('| %s | %d | %.2f | %.2f | %.2f | %s | %s |' % (s['tag'], s['pol'], s['mk'], s['sl'][0], s['sl'][1], ident, s['eng'].replace('n3_sched: ', '')))
print('\nReference = %s. makespan = last request end - t_run0 (slot 1 starts 25 s after slot 0). Divergence after the first differing token cascades into later prompts.' % (ref['tag'] if ref else 'n/a'))
