#!/usr/bin/env python3
"""L0 summarizer. usage: summarize.py <run dir> [idle_s=2.0]. Reads <run>/*/ledger/*.csv and *.csv.tasks (LLAMA_N3_LEDGER output).
wall_ms = time inside the decode call (async prefill batches return before GPU finishes). attributed_ms = start-to-start period to the NEXT decode
(covers GPU time of an async batch + host gap); periods > idle_s are counted as idle (think time / turn boundary), not as wall."""
import sys, glob, csv, os, collections
run = sys.argv[1]; IDLE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
def rd(p):
    with open(p) as f: return list(csv.DictReader(f))
files = sorted(glob.glob(run + '/*/ledger/*.csv'))
if not files: sys.exit('no ledger csv under ' + run)
def f(r, k): return float(r[k])
def summarize(rows, tasks, label):
    rows = [r for r in rows if r.get('ret', '0') == '0']
    rows.sort(key=lambda r: f(r, 't_start'))
    out = ['### ' + label, '']
    n = len(rows)
    if n == 0: return out + ['no decodes', '']
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < n else None
        per = (f(nxt, 't_start') - f(r, 't_start')) * 1000 if nxt else float('nan')
        r['_att'] = per if (nxt and per < IDLE * 1000) else None   # None = idle/last
        g = f(r, 'gap_ms'); r['_gap'] = g if 0 <= g < IDLE * 1000 else None
        r['_idle'] = g if g >= IDLE * 1000 else 0.0
        r['_n'] = int(r['n_tokens']); r['_act'] = int(r['n_active']); r['_np'] = int(r['n_prompt']); r['_ng'] = int(r['n_gen'])
    t0, t1 = f(rows[0], 't_start'), f(rows[-1], 't_start')
    tot_wall = sum(f(r, 'wall_ms') for r in rows); tot_gap = sum(r['_gap'] or 0 for r in rows); tot_idle = sum(r['_idle'] for r in rows)
    tot_att = sum(r['_att'] or 0 for r in rows)
    out += ['decodes %d, span %.1f s, sum call wall %.1f s, sum attributed %.1f s, sum host gap (<%.0fs) %.1f s, idle (>= %.0fs) %.1f s' % (n, t1 - t0, tot_wall / 1e3, tot_att / 1e3, IDLE, tot_gap / 1e3, IDLE, tot_idle / 1e3), '',
            '| kind | decodes | tokens | sum call wall s | mean call wall ms | sum attributed s | mean attributed ms | mean gap ms | sum gap s |', '|---|---|---|---|---|---|---|---|---|']
    for k in ('prefill', 'decode', 'mixed'):
        R = [r for r in rows if r['kind'] == k]
        if not R: out.append('| %s | 0 | 0 | - | - | - | - | - | - |' % k); continue
        A = [r['_att'] for r in R if r['_att'] is not None]; G = [r['_gap'] for r in R if r['_gap'] is not None]
        out.append('| %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (k, len(R), sum(r['_n'] for r in R), sum(f(r, 'wall_ms') for r in R) / 1e3, sum(f(r, 'wall_ms') for r in R) / len(R),
              sum(A) / 1e3, (sum(A) / len(A)) if A else float('nan'), (sum(G) / len(G)) if G else float('nan'), sum(G) / 1e3))
    mixed = [r for r in rows if r['kind'] == 'mixed']
    out += ['', 'mixed-batch share: %d / %d decodes (%.2f%%), %d / %d tokens (%.2f%%)' % (len(mixed), n, 100.0 * len(mixed) / n, sum(r['_n'] for r in mixed), sum(r['_n'] for r in rows), 100.0 * sum(r['_n'] for r in mixed) / max(1, sum(r['_n'] for r in rows)))]
    cob = [r for r in mixed if r['_ng'] > 0 and r['_np'] > 0]
    out.append('decode rows co-batched with a prefill chunk: %s (%d batches; max decode rows in such a batch %d)' % ('YES' if cob else 'NO', len(cob), max([r['_ng'] for r in cob], default=0)))
    # prefill with the other slot generating = decode of the other slot stalled behind the chunk
    pf = [r for r in rows if r['kind'] == 'prefill']
    out += ['', 'prefill throughput by active slots (tok / attributed time; call-wall variant in brackets):', '', '| active slots | batches | tokens | tok/s attributed | tok/s call-wall |', '|---|---|---|---|---|']
    for a in (1, 2):
        R = [r for r in pf if r['_act'] == a and r['_att'] is not None]
        if not R: out.append('| %d | 0 | 0 | - | - |' % a); continue
        tk = sum(r['_n'] for r in R); ta = sum(r['_att'] for r in R) / 1e3; tw = sum(f(r, 'wall_ms') for r in R) / 1e3
        out.append('| %d | %d | %d | %.1f | %.1f |' % (a, len(R), tk, tk / ta if ta else float('nan'), tk / tw if tw else float('nan')))
    out += ['', 'row split: slot0 %d, slot1 %d, other %d; decode-kind rows per batch mean %.2f' % (sum(int(r['slot0']) for r in rows), sum(int(r['slot1']) for r in rows), sum(int(r['slot_other']) for r in rows),
            (sum(r['_n'] for r in rows if r['kind'] == 'decode') / max(1, sum(1 for r in rows if r['kind'] == 'decode'))))]
    if tasks:
        S = lambda k: sum(int(float(t[k])) for t in tasks)
        np_, pre, app = S('n_prompt_tokens'), S('n_prefill'), S('n_appended')
        out += ['', '| tasks | prompt tok total | reused from cache (n_past) | prefilled | appended (new vs slot cache) | prefilled but not new (recompute) | ckpt restores | full resets |', '|---|---|---|---|---|---|---|---|',
                '| %d | %d | %d | %d | %d | %d | %d | %d |' % (len(tasks), np_, S('n_past'), pre, app, pre - app, S('ckpt_restore'), S('full_reset'))]
    return out + ['']
res = ['# L0 ledger summary: ' + os.path.basename(run.rstrip('/')), '']
allrows, alltasks = [], []
for p in files:
    rows = rd(p); tp = p + '.tasks'; tasks = rd(tp) if os.path.exists(tp) else []
    allrows += rows; alltasks += tasks
    res += summarize(rows, tasks, os.path.relpath(p, run))
res += summarize(allrows, alltasks, 'ALL (decodes pooled across servers)')
print('\n'.join(res))
