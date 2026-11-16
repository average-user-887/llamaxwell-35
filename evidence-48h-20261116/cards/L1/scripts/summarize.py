#!/usr/bin/env python3
"""L1 summarizer: split the host gap after each decode into server-side stages. usage: summarize.py <run dir> [idle_s=2.0]
Reads <run>/*/ledger/<name>.csv (decodes), .csv.tasks, .csv.stages. Gap window = (decode end, next decode start), decode end = t_start + wall_ms.
Stage overlap with the window is summed; nested stages (slot_launch inside task_proc) are shown but excluded from the total.
remainder = batch build / slot loop / metrics (not separately timed). Windows >= idle_s are idle: classified client-side vs server-side by http_recv."""
import sys, glob, csv, os, collections
run = sys.argv[1]; IDLE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
def rd(p): return list(csv.DictReader(open(p))) if os.path.exists(p) else []
TOP = ['http_recv_to_post', 'tokenise', 'post_tasks', 'task_proc', 'cache_match_ckpt', 'ckpt_create', 'post_decode']
NEST = ['slot_launch']
def ov(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0))
def one(p, label):
    D = [r for r in rd(p) if r.get('ret', '0') == '0']; D.sort(key=lambda r: float(r['t_start']))
    ST = rd(p + '.stages'); out = ['### ' + label, '']
    if not D: return out + ['no decodes', ''], None
    ev = collections.defaultdict(list)
    for e in ST:
        t = float(e['t_start']); ev[e['stage']].append((t, t + float(e['dur_ms']) / 1e3, e))
    win = []
    for i in range(len(D) - 1):
        e0 = float(D[i]['t_start']) + float(D[i]['wall_ms']) / 1e3; s1 = float(D[i + 1]['t_start'])
        win.append((D[i]['kind'], e0, s1))
    acc = {k: collections.defaultdict(float) for k in ('prefill', 'decode', 'mixed')}
    cnt = collections.Counter(); gsum = collections.defaultdict(float); idle = []
    for k, e0, s1 in win:
        g = s1 - e0
        if g >= IDLE: idle.append((k, e0, s1)); continue
        if g < 0: g = 0
        cnt[k] += 1; gsum[k] += g
        tot = 0.0
        for st in TOP[1:] + NEST:
            o = sum(ov(e0, s1, a, b) for a, b, _ in ev.get(st, []))
            acc[k][st] += o
            if st in TOP: tot += o
        acc[k]['sse_result_n'] += sum(1 for a, b, _ in ev.get('sse_result', []) if e0 <= a <= s1)
        acc[k]['remainder'] += max(0.0, g - tot)
    out += ['| after decode kind | windows | sum gap s | ' + ' | '.join(s + ' s' for s in TOP[1:]) + ' | slot_launch s (nested) | remainder (batch build etc) s | sse_result events |', '|' + '---|' * (6 + len(TOP))]
    for k in ('prefill', 'decode', 'mixed'):
        a = acc[k]
        out.append('| %s | %d | %.3f | %s | %.3f | %.3f | %d |' % (k, cnt[k], gsum[k], ' | '.join('%.3f' % a[s] for s in TOP[1:]), a['slot_launch'], a['remainder'], a['sse_result_n']))
    # worst prefill windows: which stage dominates
    big = []
    for k, e0, s1 in win:
        g = s1 - e0
        if k == 'prefill' and 0.05 < g < IDLE:
            d = {st: sum(ov(e0, s1, a, b) for a, b, _ in ev.get(st, [])) for st in TOP[1:]}
            big.append((g, max(d, key=d.get) if d else 'none', d))
    if big:
        c = collections.Counter(b[1] for b in big)
        out += ['', 'prefill-only windows > 50 ms: %d, dominant stage counts: %s' % (len(big), dict(c))]
    # idle classification
    tl = {}  # task -> slot
    for a, b, e in ev.get('slot_launch', []): tl[e['task']] = int(e['slot'])
    recv = sorted((a, int(e['aux'])) for a, b, e in ev.get('http_recv', []))
    out += ['', 'idle windows (>= %.1f s between decodes): %d, total %.1f s' % (IDLE, len(idle), sum(s1 - e0 for _, e0, s1 in idle))]
    cw = sw = 0.0; nr = 0
    for k, e0, s1 in idle:
        r = [t for t, _ in recv if e0 <= t <= s1]
        if r:
            nr += 1; cw += r[0] - e0; sw += s1 - r[0]
    out.append('idle windows containing a request arrival: %d; client-side wait (decode end -> http_recv) %.1f s; server-side (http_recv -> next decode start) %.1f s' % (nr, cw, sw))
    fin = collections.defaultdict(list)
    for a, b, e in ev.get('sse_result', []):
        if e['aux'] == '1' and e['task'] in tl: fin[tl[e['task']]].append(a)
    lag = []
    for t, sl in recv:
        pr = [f for f in fin.get(sl, []) if f <= t]
        if pr: lag.append(t - max(pr))
    if lag: out.append('request arrival vs previous final response on the same slot (client think time): n=%d, mean %.2f s, min %.2f, max %.2f, sum %.1f s' % (len(lag), sum(lag) / len(lag), min(lag), max(lag), sum(lag)))
    ts = [float(e['t_start']) for e in ST if e['stage'] == 'tokenise']
    for st in ('tokenise', 'post_tasks', 'task_proc', 'slot_launch', 'cache_match_ckpt', 'ckpt_create', 'post_decode'):
        v = [float(e['dur_ms']) for e in ST if e['stage'] == st]
        if v: out.append('stage %s: n=%d sum %.3f s mean %.2f ms max %.1f ms' % (st, len(v), sum(v) / 1e3, sum(v) / len(v), max(v)))
    return out + [''], (cnt, gsum, acc, len(idle), sw, cw)
res = ['# L1 stage split: ' + os.path.basename(run.rstrip('/')), '']
files = sorted(glob.glob(run + '/*/ledger/*.csv'))
if not files: sys.exit('no ledger csv under ' + run)
agg = collections.defaultdict(float)
for p in files:
    o, a = one(p, os.path.relpath(p, run)); res += o
    if a:
        cnt, gsum, acc, ni, sw, cw = a
        for k in ('prefill', 'decode', 'mixed'):
            agg[k + ':gap'] += gsum[k]
            for s in TOP[1:] + NEST + ['remainder']: agg[k + ':' + s] += acc[k][s]
res += ['## Totals over all ledger files (gap after prefill-only decodes, per replay sums shown above)', '', '| stage | sum s after prefill-only | sum s after decode-only | sum s after mixed |', '|---|---|---|---|']
for s in ['gap'] + TOP[1:] + NEST + ['remainder']:
    res.append('| %s | %.3f | %.3f | %.3f |' % (s, agg['prefill:' + s], agg['decode:' + s], agg['mixed:' + s]))
print('\n'.join(res))
