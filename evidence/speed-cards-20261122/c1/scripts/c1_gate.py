#!/usr/bin/env python3
"""C1 gate v2: two-slot MTP ON vs OFF from <dir>/<n>_<arm>/n3r/run.json (+ <dir>/results.json). Prints markdown.
Drops invalid runs (invalid / tmax>=80 / ok!=True / rc!=0). Makespan raw and pause-subtracted (pauses overlapping the
timed phase). Decode = tokens / decode-seconds summed over turns (all, and non-void only). Flags runs with <1920 tokens
(12 turns x 160) or any early stop. Usage: c1_gate.py DIR [DIR...]"""
import json, statistics as st, sys
from pathlib import Path
FULL = 1920
runs, order = {}, []
for a in sys.argv[1:]:
    D = Path(a)
    res = {}
    if (D / 'results.json').exists():
        for r in json.loads((D / 'results.json').read_text()):
            res['%s_%s' % (r['run'], r['arm'])] = r
    for rd in sorted(D.glob('*_*'), key=lambda p: int(p.name.split('_')[0]) if p.name.split('_')[0].isdigit() else 999):
        if not rd.is_dir(): continue
        arm = rd.name.split('_', 1)[1]; meta = res.get(rd.name, {}); j = rd / 'n3r' / 'run.json'
        order.append('%s/%s' % (D.name, rd.name))
        if not j.exists():
            print('DROP %s/%s: no run.json (rc=%s oom=%s)' % (D.name, rd.name, meta.get('rc'), bool(meta.get('oom')))); continue
        R = json.loads(j.read_text()); T = R['turns_rec']
        why = [w for w, c in (('invalid', meta.get('invalid')), ('tmax>=80', (meta.get('tmax') or 0) >= 80),
                              ('ok!=True', R.get('ok') is not True), ('rc!=0', meta.get('rc', 0) != 0)) if c]
        if why: print('DROP %s/%s: %s' % (D.name, rd.name, ','.join(why))); continue
        t0, t1 = R['t_run0'], R['t_run0'] + R['makespan_s']
        pin = sum(max(0, min(p0 + d, t1) - max(p0, t0)) for p0, d in (meta.get('pauses') or []))
        ntok = sum(t.get('tokens_predicted') or 0 for t in T)
        early = [(t['agent'], t['turn'], t['tokens_predicted'], t['stop_type']) for t in T if (t.get('tokens_predicted') or 0) < t.get('n_cap', 160)]
        nv = [t for t in T if not t.get('void')]
        tps = lambda L: sum(t['tokens_predicted'] for t in L) / (sum(t['predicted_ms'] for t in L) / 1000) if L else None
        runs.setdefault(arm, []).append(dict(dir='%s/%s' % (D.name, rd.name), mk=R['makespan_s'], mkp=R['makespan_s'] - pin, pin=pin,
            void=R['n_void'], ntok=ntok, early=early, agg=tps(T), agg_nv=tps(nv), nv={(t['agent'], t['turn']) for t in nv},
            T={(t['agent'], t['turn']): t for t in T}, acc=sum(t.get('draft_n_accepted') or 0 for t in T),
            drf=sum(t.get('draft_n') or 0 for t in T), tmax=meta.get('tmax'), vram=meta.get('vram_peak_mib'), sha=R['run_sha'][:12]))
print('\nrun order: ' + ', '.join(order))
print('\n| run | makespan s | pause in timed s | makespan-pause s | tokens | tok/decode-s all | non-void | draft acc | void | tmax | VRAM MiB | flag |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for arm in sorted(runs):
    for r in runs[arm]:
        fl = 'SHORT %d<%d early=%s' % (r['ntok'], FULL, r['early']) if r['ntok'] < FULL else ''
        print('| %s | %.2f | %.1f | %.2f | %d | %.1f | %s | %d/%d | %d | %s | %s | %s |' % (r['dir'], r['mk'], r['pin'], r['mkp'], r['ntok'],
              r['agg'], '%.1f' % r['agg_nv'] if r['agg_nv'] else '-', r['acc'], r['drf'], r['void'], r['tmax'], r['vram'], fl))
m = lambda L, k: st.mean(r[k] for r in L)
for c in ('64', '78', '96'):
    for full in (False, True):
        on, off = runs.get('ON' + c, []), runs.get('OFF' + c, [])
        if full: on, off = [r for r in on if r['ntok'] >= FULL], [r for r in off if r['ntok'] >= FULL]
        if not on or not off: continue
        common = set.intersection(*[r['nv'] for r in on + off])
        ptps = lambda L: sum(r['T'][k]['tokens_predicted'] for r in L for k in common) / (sum(r['T'][k]['predicted_ms'] for r in L for k in common) / 1000) if common else None
        pon, poff = ptps(on), ptps(off)
        print('\n%sK%s: n ON=%d OFF=%d | makespan raw ON %.2f OFF %.2f -> %+.2f%% | pause-subtracted ON %.2f OFF %.2f -> %+.2f%% | tok/decode-s all ON %.1f OFF %.1f -> %+.1f%% | paired non-void turns %d: ON %s OFF %s -> %s' % (
            c, ' (full-token runs only)' if full else '', len(on), len(off), m(on, 'mk'), m(off, 'mk'), 100 * (m(off, 'mk') / m(on, 'mk') - 1),
            m(on, 'mkp'), m(off, 'mkp'), 100 * (m(off, 'mkp') / m(on, 'mkp') - 1), m(on, 'agg'), m(off, 'agg'), 100 * (m(on, 'agg') / m(off, 'agg') - 1),
            len(common), '%.1f' % pon if pon else '-', '%.1f' % poff if poff else '-', '%+.1f%%' % (100 * (pon / poff - 1)) if pon and poff else '-'))
print('\nlossless: not judgeable from two-slot runs (OFF-vs-OFF turn token_sha also differs); needs a fixed-schedule single-slot greedy ON/OFF check.')
