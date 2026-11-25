#!/usr/bin/env python3
"""Score the C3 chain dir -> <chain>/../../RESULT.md (table first). usage: c3_score.py /tmp/qa-speed-cards/c3/runs/chain"""
import json, math, sys, time
from pathlib import Path
B = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/qa-speed-cards/c3/runs/chain')
OUT = Path('/tmp/qa-speed-cards/c3/RESULT.md')


def load(stage):
    f = B / stage / 'results.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.is_file() else []


def ok(r):
    return not r.get('error') and not r.get('invalid') and not r.get('skipped')


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else None


def pct(a, b):
    return '%+.2f%%' % (100 * (a / b - 1)) if a and b else 'n/a'


def dist(e):
    """one position -> {token_key: logprob}"""
    for k, lp in (('top_logprobs', True), ('top_probs', False), ('probs', False)):
        if isinstance(e.get(k), list) and e[k]:
            d = {}
            for t in e[k]:
                key = t.get('id', t.get('token', t.get('tok_str')))
                v = t.get('logprob') if lp else t.get('prob')
                if v is None: continue
                d[key] = v if lp else math.log(max(v, 1e-12))
            return d
    return {}


def positions(stage, tag):
    f = B / stage / tag / 'probs.json'
    if not f.is_file(): return []
    out = []
    for group in json.loads(f.read_text()):
        for e in (group if isinstance(group, list) else []):
            out.append(dist(e))
    return out


def lp_proxy(stage, ta, tb):
    pa, pb = positions(stage, ta), positions(stage, tb)
    n = min(len(pa), len(pb)); kl = []; agree = 0; d1 = []
    for a, b in zip(pa[:n], pb[:n]):
        if not a or not b: continue
        zb = math.log(sum(math.exp(v) for v in b.values())); floor = min(a.values())
        kl.append(sum(math.exp(v - zb) * ((v - zb) - (a.get(k, floor) - zb)) for k, v in b.items()))
        tb1 = max(b, key=b.get); ta1 = max(a, key=a.get); agree += tb1 == ta1
        d1.append(a.get(tb1, floor) - b[tb1])
    if not kl: return None
    return dict(n=len(kl), kld_proxy_mean=sum(kl) / len(kl), top1_agree=agree / len(kl), top1_dlogp_mean=sum(d1) / len(d1))


def main():
    L = []
    abba, slot2, sweep, b0 = load('abba'), load('slot2'), load('sweep1'), load('b0')
    # gate table first
    A = [r for r in abba if r['tag'].startswith('A') and ok(r)]; Bb = [r for r in abba if r['tag'].startswith('B') and ok(r)]
    best = json.loads((B / 'best.json').read_text()) if (B / 'best.json').is_file() else {}
    L += ['# C3 RESULT - Maxwell FA GEMM tile/split tuning (append prefill @80K)', '',
          '_Updated %s CET. Chain dir runs/chain. Driver log runs/chain/driver.log._' % time.strftime('%H:%M'), '',
          '## Gate table', '', '| check | A (v1 + %s) | B (B0 card1) | delta | gate | status |' % (best.get('tag', '?')), '|---|---|---|---|---|---|']
    aa, bb = mean([r['append_tok_s'] for r in A]), mean([r['append_tok_s'] for r in Bb])
    L.append('| append prefill tok/s @80-88K (ABBA n=%d/%d) | %s | %s | %s | >= +3%% | %s |' % (
        len(A), len(Bb), '%.1f' % aa if aa else '-', '%.1f' % bb if bb else '-', pct(aa, bb),
        ('PASS' if aa / bb - 1 >= 0.03 else 'FAIL') if aa and bb and len(A) >= 2 and len(Bb) >= 2 else 'pending'))
    da = mean([r.get('decode', {}).get('predicted_per_second') for r in A]); db = mean([r.get('decode', {}).get('predicted_per_second') for r in Bb])
    L.append('| 1-slot decode tok/s @88K | %s | %s | %s | no regression (> -1%%) | %s |' % (
        '%.2f' % da if da else '-', '%.2f' % db if db else '-', pct(da, db), ('PASS' if da / db - 1 > -0.01 else 'FAIL') if da and db else 'pending'))
    A2 = [r for r in slot2 if r['tag'].startswith('A') and ok(r)]; B2 = [r for r in slot2 if r['tag'].startswith('B') and ok(r)]
    a2, b2 = mean([r.get('agg_tok_s') for r in A2]), mean([r.get('agg_tok_s') for r in B2])
    L.append('| 2-slot aggregate append tok/s @40K (n=%d/%d) | %s | %s | %s | no regression (> -1%%) | %s |' % (
        len(A2), len(B2), '%.1f' % a2 if a2 else '-', '%.1f' % b2 if b2 else '-', pct(a2, b2), ('PASS' if a2 / b2 - 1 > -0.01 else 'FAIL') if a2 and b2 else 'pending'))
    lps = []
    tags = [r['tag'] for r in abba if ok(r)]
    for i in range(0, 8, 2):
        pair = [t for t in tags if t[1:] in (str(i), str(i + 1))]
        ta = [t for t in pair if t[0] == 'A']; tb = [t for t in pair if t[0] == 'B']
        if ta and tb:
            p = lp_proxy('abba', ta[0], tb[0])
            if p: lps.append(p)
    if lps:
        km = mean([p['kld_proxy_mean'] for p in lps]); ag = mean([p['top1_agree'] for p in lps])
        L.append('| logprob proxy (top-8, PROXY not KLD), mean KL(B||A) over %d positions | top1 agree %.3f | ref | %.5f | <= +.0010 | %s |' % (
            sum(p['n'] for p in lps), ag, km, 'PASS(proxy)' if km <= 0.001 else 'FAIL(proxy)'))
    else:
        L.append('| logprob proxy (top-8, PROXY not KLD) | - | - | - | <= +.0010 | pending |')
    L += ['', 'FP64 KLD harness: none runnable for an 80K-deep append in this chain; the row above is a labelled top-8 logprob proxy '
          '(3 append next-token positions + 48 decode positions per arm, A vs adjacent B in the same ABBA block).', '']
    # breakdown
    bf = B / 'profile' / 'breakdown.json'
    L += ['## Step 1 breakdown (GGML_CUDA_NODE_TIMING, prefill phase; per-node sync, shares only)', '']
    if bf.is_file():
        s = json.loads(bf.read_text()); t = s['total_ms']
        L += ['| class | ms | share |', '|---|---|---|'] + ['| %s | %.0f | %.1f%% |' % (k, v, 100 * v / t) for k, v in s['ms'].items()]
        L += ['', 'Measured attention (FLASH_ATTN_EXT) share over setup+appends: %.1f%%; projected to the 80-88K append window: %.1f%%. Gate >=15%%.' % (
            100 * s['share_measured'], 100 * s['share_append_est']), '', 'Top ops: ' + ', '.join('%s %s ms' % (o, m) for o, m in s['top']), '']
    else:
        L += ['pending', '']
    L += ['## Sweep 1 (v1 lib, env knobs)', '', '| arm | env | append tok/s | vs v1-default | decode tok/s | tmax C | valid | err |', '|---|---|---|---|---|---|---|---|']
    ctl = mean([r.get('append_tok_s') for r in sweep if r['tag'] in ('d', 'd2') and ok(r)])
    for r in b0 + sweep:
        L.append('| %s%s | %s | %s | %s | %s | %s | %s | %s |' % (
            'B0 ' if r in b0 else '', r['tag'], r.get('env_over') or '{}', r.get('append_tok_s'), pct(r.get('append_tok_s'), ctl) if ctl else '-',
            (r.get('decode') or {}).get('predicted_per_second'), r.get('tmax'), ok(r), (r.get('error') or '')[:60]))
    L += ['', '## ABBA arms', '', '| arm | append tok/s | decode tok/s | tmax | valid |', '|---|---|---|---|---|']
    for r in abba:
        L.append('| %s | %s | %s | %s | %s |' % (r['tag'], r.get('append_tok_s'), (r.get('decode') or {}).get('predicted_per_second'), r.get('tmax'), ok(r)))
    L += ['', '## 2-slot arms', '', '| arm | agg tok/s | tmax | valid | err |', '|---|---|---|---|---|']
    for r in slot2:
        L.append('| %s | %s | %s | %s | %s |' % (r['tag'], r.get('agg_tok_s'), r.get('tmax'), ok(r), (r.get('error') or '')[:60]))
    dl = B / 'driver.log'
    if dl.is_file():
        tail = [l for l in dl.read_text().splitlines() if 'GATE' in l or 'KILL' in l or 'sweep1 vs' in l or 'chain complete' in l or 'crash' in l]
        L += ['', '## Driver decisions', ''] + ['- ' + l for l in tail]
    OUT.write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
