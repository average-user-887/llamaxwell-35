#!/usr/bin/env python3
"""N3 analyzer. usage: n3_analyze.py <rundir> | --selftest
Reads <rundir>/runs/*/run.json (+telemetry.csv); writes <rundir>/N3_TABLE.md and n3_summary.json.
Predeclared: paired per-block difference (arm - b3, same block) as % of b3 value; CI95 = t(n-1)*sd/sqrt(n) over blocks (5); a difference counts only if CI excludes 0.
cap1024 'worth it' iff mean makespan_total cost <= +2% AND mean max-gap cut >= 40% (both counted only if CI excludes 0 for the gap cut)."""
import sys, os, json, glob, math, statistics as st, tempfile, random
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
def ci(xs):
    n = len(xs); m = st.mean(xs)
    if n < 2: return m, None, None
    h = T975.get(n - 1, 2.0) * st.stdev(xs) / math.sqrt(n); return m, m - h, m + h
def telem(path, a, b):
    rows = []
    try:
        for l in open(path).read().splitlines()[1:]:
            p = l.split(',')
            try: rows.append(tuple(float(x) for x in p[:5]))
            except ValueError: pass
    except OSError: return {}
    w = [r for r in rows if a - 1 <= r[0] <= b + 1]
    if not w: return {}
    sm = [r[1] for r in w]; pw = [r[3] for r in w]
    return {'vram_peak_mib': max(r[4] for r in rows), 'sm_mean_mhz': st.mean(sm), 'sm_min_mhz': min(sm), 'power_mean_w': st.mean(pw), 'power_max_w': max(pw), 'n_samples': len(w)}
def metrics(R, d):
    q = [r for r in R.get('reqs', []) if r.get('t_end')]; M = {}
    if not q: return M
    t0 = min(r['t_send'] for r in q); t1 = max(r['t_end'] for r in q); M['makespan_total_s'] = t1 - t0
    per = {}
    for a in (0, 1):
        x = [r for r in q if r['agent'] == a]
        if x: per[a] = (min(r['t_send'] for r in x), max(r['t_end'] for r in x)); M['makespan_a%d_s' % a] = per[a][1] - per[a][0]
    for r in q:
        M['turn_a%d_t%d_s' % (r['agent'], r['turn'])] = r['t_end'] - r['t_send']
    tt = [r['t_first'] - r['t_send'] for r in q if r.get('t_first')]; M['ttft_mean_s'] = st.mean(tt) if tt else None
    for k in ('1K', '4K'):
        x = [r['t_first'] - r['t_send'] for r in q if r['kind'] == k and r.get('t_first')]
        if x: M['ttft_%s_mean_s' % k] = st.mean(x)
    gaps = {}
    for r in q:
        ts = [e[0] for e in r.get('ev', [])]; g = [b - a for a, b in zip(ts, ts[1:])]
        if g: gaps[r['agent']] = max(gaps.get(r['agent'], 0), max(g))
    if gaps: M['max_gap_s'] = max(gaps.values()); [M.__setitem__('max_gap_a%d_s' % a, v) for a, v in gaps.items()]
    M['prefill_s'] = sum((r.get('prompt_ms') or 0) for r in q) / 1000; M['decode_s'] = sum((r.get('predicted_ms') or 0) for r in q) / 1000
    M['restore_s'] = sum(x['s'] for x in R.get('restore', [])); M['prefill_tokens'] = sum((r.get('prompt_n') or 0) for r in q)
    M['decode_tokens'] = sum((r.get('tokens_predicted') or 0) for r in q); M['gen_tps_overall'] = M['decode_tokens'] / M['makespan_total_s']
    if len(per) == 2:   # common window = both agents active
        cs = max(per[0][0], per[1][0]); ce = min(per[0][1], per[1][1]); M['common_window_s'] = max(0, ce - cs)
        if ce > cs:
            nt = sum(n for r in q for t, n in r.get('ev', []) if cs <= t <= ce); M['cw_decode_tps'] = nt / (ce - cs)
            M['cw_prefill_tps'] = None
    M.update(telem(d + '/telemetry.csv', t0, t1)); M['valid'] = 1.0 if R.get('ok') else 0.0
    M['id_gap_total'] = sum((r.get('id_gap') or 0) for r in q)
    return M
def load(rd):
    runs = []
    for f in sorted(glob.glob(rd + '/runs/*/run.json')):
        R = json.load(open(f)); 
        if R.get('tag') == 'slotbuild': continue
        R['M'] = metrics(R, os.path.dirname(f)); runs.append(R)
    return runs
def paired(runs, arm, metric, base='b3'):
    out = []
    for blk in sorted({r['block'] for r in runs if r['kind'] == 'main'}):
        a = [r for r in runs if r['kind'] == 'main' and r['block'] == blk and r['arm'] == arm and r['M'].get(metric) is not None]
        b = [r for r in runs if r['kind'] == 'main' and r['block'] == blk and r['arm'] == base and r['M'].get(metric) is not None]
        if a and b and b[0]['M'][metric]: out.append((blk, a[0]['M'][metric] - b[0]['M'][metric], 100 * (a[0]['M'][metric] / b[0]['M'][metric] - 1)))
    return out
def f(x, p=2): return '-' if x is None else ('%.*f' % (p, x))
PRIMARY = [('makespan_total_s', 'makespan total s'), ('makespan_a0_s', 'makespan agent0 s'), ('makespan_a1_s', 'makespan agent1 s'), ('ttft_mean_s', 'TTFT mean s'), ('ttft_4K_mean_s', 'TTFT 4K turns s'),
           ('ttft_1K_mean_s', 'TTFT 1K turns s'), ('max_gap_s', 'max token gap s'), ('prefill_s', 'prefill s (server)'), ('decode_s', 'decode s (server)'), ('restore_s', 'restore s'),
           ('cw_decode_tps', 'common-window decode tok/s'), ('vram_peak_mib', 'peak VRAM MiB'), ('sm_mean_mhz', 'SM clock mean MHz'), ('power_mean_w', 'power mean W'), ('power_max_w', 'power max W')]
def report(rd):
    runs = load(rd); main = [r for r in runs if r['kind'] == 'main']; arms = [a for a in ('b3', 'gdn4', 'cap1024', 'gemm', 'full', 'prod') if any(r['arm'] == a for r in main)]
    S = {'n_runs': len(runs), 'arms': {}, 'paired': {}, 'verdict': {}}; L = []
    L.append('# N3 table (generated by n3_analyze.py)\n')
    inv = [r['tag'] for r in runs if not r.get('ok') and r['kind'] != 'sweep']; L.append('Runs: %d (main %d); not-ok runs (receipt or error): %s\n' % (len(runs), len(main), inv or 'none'))
    L.append('## Per arm: mean [min,max] over blocks (n)\n'); L.append('| metric | ' + ' | '.join(arms) + ' |'); L.append('|---|' + '---|' * len(arms))
    for k, lab in PRIMARY:
        row = []
        for a in arms:
            v = [r['M'][k] for r in main if r['arm'] == a and r['M'].get(k) is not None]; S['arms'].setdefault(a, {})[k] = v
            row.append('%s [%s,%s] (%d)' % (f(st.mean(v)), f(min(v)), f(max(v)), len(v)) if v else '-')
        L.append('| %s | ' % lab + ' | '.join(row) + ' |')
    L.append('\n## Paired difference vs b3 (same block), % of b3; mean +/- CI95 half-width (t, n blocks). * = CI excludes 0\n')
    others = [a for a in arms if a != 'b3']; L.append('| metric | ' + ' | '.join(others) + ' |'); L.append('|---|' + '---|' * len(others))
    for k, lab in PRIMARY:
        row = []
        for a in others:
            p = paired(runs, a, k)
            if len(p) >= 2:
                m, lo, hi = ci([x[2] for x in p]); ma, _, _ = ci([x[1] for x in p]); sig = (lo > 0 or hi < 0)
                S['paired'].setdefault(a, {})[k] = {'n': len(p), 'mean_abs': ma, 'mean_pct': m, 'ci95_pct': [lo, hi], 'sig': sig}
                row.append('%s%% +/- %s (abs %s)%s n=%d' % (f(m), f((hi - lo) / 2), f(ma), ' *' if sig else '', len(p)))
            else: row.append('n<2')
        L.append('| %s | ' % lab + ' | '.join(row) + ' |')
    L.append('\n## Per-turn duration s (arrival to finish), mean over blocks\n'); 
    turns = sorted({k for r in main for k in r['M'] if k.startswith('turn_')}); L.append('| turn | ' + ' | '.join(arms) + ' |'); L.append('|---|' + '---|' * len(arms))
    for k in turns:
        L.append('| %s | ' % k[5:] + ' | '.join((lambda v: f(st.mean(v)) if v else '-')([r['M'][k] for r in main if r['arm'] == a and k in r['M']]) for a in arms) + ' |')
    if 'cap1024' in S['paired']:
        pm = S['paired']['cap1024'].get('makespan_total_s'); pg = S['paired']['cap1024'].get('max_gap_s')
        if pm and pg:
            cost = pm['mean_pct']; cut = -pg['mean_pct']; gap_sig = pg['sig']; ok = cost <= 2.0 and cut >= 40.0 and gap_sig
            S['verdict']['cap1024'] = {'makespan_cost_pct': cost, 'makespan_cost_ci95': pm['ci95_pct'], 'makespan_cost_sig': pm['sig'], 'gap_cut_pct': cut, 'gap_cut_ci95': [-pg['ci95_pct'][1], -pg['ci95_pct'][0]], 'gap_sig': gap_sig, 'worth_it': ok}
            L.append('\n## cap1024 verdict (predeclared: cost <= +2%% makespan AND gap cut >= 40%%, CI excludes 0)\n\nmakespan cost %s%% (CI %s..%s), max-gap cut %s%% (CI %s..%s) -> **%s**' % (
                f(cost), f(pm['ci95_pct'][0]), f(pm['ci95_pct'][1]), f(cut), f(-pg['ci95_pct'][1]), f(-pg['ci95_pct'][0]), 'WORTH IT' if ok else 'NOT worth it'))
    sw = [r for r in runs if r['kind'] == 'sweep']
    if sw:
        L.append('\n## N3b ubatch sweep (b3, short deep-append fixture), mean over reps\n'); L.append('| ub | reps | makespan s | TTFT s | max gap s | prefill s | peak VRAM MiB | ok |'); L.append('|---|---|---|---|---|---|---|---|')
        for ub in sorted({r['ub'] for r in sw}):
            x = [r for r in sw if r['ub'] == ub]; g = lambda k: (lambda v: f(st.mean(v)) if v else '-')([r['M'][k] for r in x if r['M'].get(k) is not None])
            L.append('| %d | %d | %s | %s | %s | %s | %s | %s |' % (ub, len(x), g('makespan_total_s'), g('ttft_mean_s'), g('max_gap_s'), g('prefill_s'), g('vram_peak_mib'), sum(1 for r in x if r.get('ok'))))
        S['sweep_failed'] = [r['tag'] + ': ' + r.get('error', '') for r in sw if r.get('error')]
        if S['sweep_failed']: L.append('\nSweep failures (e.g. OOM at -ub 2048): %s' % S['sweep_failed'])
    cf = [r for r in main if r.get('ctxfull')]
    if cf:
        L.append('\n## N3b context-full event (slot 0 pushed past %d)\n' % cf[0]['ctxfull']['n_ctx_seq']); L.append('| arm | prompt_len | http | wall s | tokens_pred | truncated | stop_type | health after | slot1 probe ok | body/err |'); L.append('|---|---|---|---|---|---|---|---|---|---|')
        for r in cf:
            c = r['ctxfull']; q = c['req']; pr = c['probe_slot1']
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (r['arm'], c['prompt_len'], q['http'], f(c['wall_s']), q['tokens_predicted'], q['truncated'], q['stop_type'], c['health_after'], pr.get('tokens_predicted') == pr.get('n_predict'), str(q.get('body') or q.get('err') or '')[:90].replace('|', '/')))
    L.append('\n## Receipt: tokens_predicted == n_predict and id_gap <= non-ASCII events, per run (id_gap total)\n')
    L.append('| run | ok | id_gap total | final lens |'); L.append('|---|---|---|---|')
    for r in runs: L.append('| %s | %s | %s | %s |' % (r['tag'], r.get('ok'), r['M'].get('id_gap_total'), r.get('final_len')))
    open(rd + '/N3_TABLE.md', 'w').write('\n'.join(L) + '\n'); json.dump(S, open(rd + '/n3_summary.json', 'w'), indent=1, default=str); print('\n'.join(L))
def selftest():
    rd = tempfile.mkdtemp(prefix='n3synth_'); random.seed(1); t = 1.0e9
    orders = [['b3', 'gdn4', 'cap1024', 'gemm']] * 5; eff = {'b3': (1.0, 1.0), 'gdn4': (1.01, 1.0), 'cap1024': (1.015, 0.5), 'gemm': (0.97, 1.0)}
    for b, order in enumerate(orders):
        for pos, arm in enumerate(order):
            tag = 'b%d_p%d_%s' % (b, pos, arm); d = '%s/runs/%s' % (rd, tag); os.makedirs(d); sl, gp = eff[arm]; reqs = []; t0 = t + 5
            for a in (0, 1):
                ts = t0 + a * 25
                for k in range(6):
                    kind = '1K' if k % 2 == 0 else '4K'; n = 256 if k % 2 == 0 else 512; pf = (7 if k % 2 == 0 else 32) * sl * random.uniform(.99, 1.01); ev = []
                    tt = ts + pf
                    for i in range(n): tt += 0.025 * sl + (4 * gp if i == 5 and kind == '4K' else 0) ; ev.append([tt, 1])
                    reqs.append({'tag': 'a%d_t%d' % (a, k), 'agent': a, 'turn': k, 'kind': kind, 't_send': ts, 't_first': ev[0][0], 't_end': tt, 'ev': ev, 'tokens_predicted': n, 'n_predict': n,
                                 'id_gap': 0, 'prompt_ms': pf * 1000, 'predicted_ms': (tt - ts - pf) * 1000, 'prompt_n': 1024, 'receipt_ok': True, 'http': 200}); ts = tt + 1
            tend = max(r['t_end'] for r in reqs); R = {'tag': tag, 'arm': arm, 'ub': 1024, 'block': b, 'kind': 'main', 'ok': True, 'reqs': reqs, 'restore': [{'s': 8}, {'s': 8}], 'final_len': {0: 95000, 1: 95000}}
            if b == 4: R['ctxfull'] = {'prompt_len': 99800, 'n_ctx_seq': 98304, 'req': {'http': 400, 'tokens_predicted': None, 'truncated': None, 'stop_type': None, 'body': '{"error":"exceeds context"}'}, 'health_after': 200, 'probe_slot1': {'tokens_predicted': 16, 'n_predict': 16}, 'wall_s': 1.2}
            json.dump(R, open(d + '/run.json', 'w'))
            with open(d + '/telemetry.csv', 'w') as fh:
                fh.write('epoch,sm_mhz,mem_mhz,power_w,mem_mib,temp_c\n'); x = t0
                while x < tend + 2: fh.write('%.2f,%d,3304,%.1f,%d,70\n' % (x, 1100 + random.randint(-20, 20), 240 + random.random() * 10, 23108), ); x += 1
            t = tend + 60
    for i, ub in enumerate((1024, 512, 2048)):
        d = '%s/runs/sw%d_ub%d' % (rd, i, ub); os.makedirs(d); json.dump({'tag': 'sw%d' % i, 'arm': 'b3', 'ub': ub, 'block': i, 'kind': 'sweep', 'ok': ub != 2048, 'error': 'container exited (OOM/launch fail)' if ub == 2048 else None, 'reqs': [], 'restore': []}, open(d + '/run.json', 'w'))
    report(rd); S = json.load(open(rd + '/n3_summary.json')); v = S['verdict']['cap1024']
    assert v['worth_it'] is False or v['worth_it'] is True; print('\nSELFTEST dir', rd, 'verdict', v)
    assert abs(v['gap_cut_pct'] - 50) < 15, v
if __name__ == '__main__':
    if sys.argv[1] == '--selftest': selftest()
    else: report(sys.argv[1])
