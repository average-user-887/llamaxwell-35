#!/usr/bin/env python3
"""H1 summarizer. usage: summarize.py <run dir>. Per session: makespan, ckpt_create / ckpt_wait_gpu / ckpt_copy sums (ledger stages), peak VRAM (5 s samples),
token identity vs the first prod session (D0 compare), engagement (H1_CKPT_ENGAGED count in server.log vs checkpoints created). Accept lines at the end."""
import sys, json, glob, re, os, csv, statistics as st
sys.path.insert(0, '<W>/D0')
import compare as C
run = sys.argv[1]; vr = []
try:
    for l in open(run + '/vram.csv'):
        p = l.strip().split(',')
        try: vr.append((float(p[0]), float(p[1])))
        except Exception: pass
except OSError: pass
def stages(tag):
    key = 'nh1-' + tag.replace('_', '-')[:24] + '-'; tot = {'ckpt_create': 0.0, 'ckpt_wait_gpu': 0.0, 'ckpt_copy': 0.0}; n = 0
    for f in glob.glob(run + '/ledger/' + key + '*.csv.stages'):
        for r in csv.DictReader(open(f)):
            if r['stage'] in tot:
                tot[r['stage']] += float(r['dur_ms']) / 1e3
                if r['stage'] == 'ckpt_create': n += 1
    return tot, n
S = []
for f in sorted(glob.glob(run + '/runs/b*/run.json')):
    R = json.load(open(f))
    if R.get('kind') != 'main' or not R.get('ok') or not R.get('reqs'): print('skipped (not ok):', f, R.get('error'), file=sys.stderr); continue
    rq = R['reqs']; mk = max(r['t_end'] for r in rq) - R['t_run0']; d = os.path.dirname(f)
    log = open(d + '/server.log', errors='replace').read()
    eng = [int(x) for x in re.findall(r'H1_CKPT_ENGAGED n=(\d+)', log)]; made = len(re.findall(r'created context checkpoint', log))
    t0, t1 = R.get('utc_start', R['t_run0']), R.get('utc_end', max(r['t_end'] for r in rq))
    pk = max([m for t, m in vr if t0 <= t <= t1], default=float('nan')); stg, n = stages(R['tag'])
    S.append(dict(tag=R['tag'], arm=R['arm'], mk=mk, peak=pk, ids=C.load(f), stg=stg, nck=n, eng=(max(eng) if eng else 0), made=made))
S.sort(key=lambda s: s['tag']); ref = next((s for s in S if s['arm'] == 'prod'), None)
def ms(v): return (st.mean(v), st.stdev(v) if len(v) > 1 else float('nan')) if v else (float('nan'),) * 2
print('| arm | n | makespan mean s | sd s | ckpt_create sum/replay s | ckpt_wait_gpu s | ckpt_copy s | peak VRAM max MiB | token-identical to prod ref | checkpoints via new path / created |\n|---|---|---|---|---|---|---|---|---|---|')
for a in ('prod', 'h1'):
    X = [s for s in S if s['arm'] == a]
    if not X: continue
    idt = sum(1 for s in X if ref and C.first_div(ref['ids'], s['ids']) is None)
    print('| %s | %d | %.2f | %.2f | %.3f | %.3f | %.3f | %.0f | %d/%d | %d / %d |' % (a, len(X), *ms([s['mk'] for s in X]), st.mean(s['stg']['ckpt_create'] for s in X), st.mean(s['stg']['ckpt_wait_gpu'] for s in X),
          st.mean(s['stg']['ckpt_copy'] for s in X), max(s['peak'] for s in X), idt, len(X), sum(s['eng'] for s in X), sum(s['made'] for s in X)))
P = [s for s in S if s['arm'] == 'prod']; H = [s for s in S if s['arm'] == 'h1']
if P and H:
    cp, ch = st.mean(s['stg']['ckpt_copy'] for s in P), st.mean(s['stg']['ckpt_copy'] for s in H)
    pp = max(s['peak'] for s in P); ph = max(s['peak'] for s in H)
    print('\nACCEPT ckpt_copy cut >= 3x (prod %.3f s -> h1 %.3f s, %.2fx): %s' % (cp, ch, cp / ch if ch else float('inf'), 'PASS' if ch and cp / ch >= 3 else 'FAIL'))
    print('ACCEPT peak VRAM h1 %.0f <= prod %.0f + 100 MiB: %s' % (ph, pp, 'PASS' if ph <= pp + 100 else 'FAIL'))
    print('ACCEPT token identity h1 == prod ref: %s' % ('PASS' if all(ref and C.first_div(ref['ids'], s['ids']) is None for s in H) else 'FAIL'))
    print('ACCEPT engagement > 0 (new-path checkpoints in h1, 0 in prod): %s' % ('PASS' if sum(s['eng'] for s in H) > 0 and sum(s['eng'] for s in P) == 0 else 'FAIL'))
print('\nPer session:\n\n| session | arm | makespan s | ckpt_create s | wait_gpu s | copy s | n ckpt | peak MiB | identity vs prod ref | engaged |\n|---|---|---|---|---|---|---|---|---|---|')
for s in S:
    dv = C.first_div(ref['ids'], s['ids']) if ref else None
    print('| %s | %s | %.2f | %.3f | %.3f | %.3f | %d | %.0f | %s | %d |' % (s['tag'], s['arm'], s['mk'], s['stg']['ckpt_create'], s['stg']['ckpt_wait_gpu'], s['stg']['ckpt_copy'], s['nck'], s['peak'], 'identical' if dv is None else str(dv), s['eng']))
print('\nmakespan = last request end - t_run0; ckpt_create = server stage (wait + copy + bookkeeping); peak = max whole-GPU 5 s sample in the session window.')
