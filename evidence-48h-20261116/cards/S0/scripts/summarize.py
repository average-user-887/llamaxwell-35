#!/usr/bin/env python3
"""S0 summarizer. usage: summarize.py <run dir>. Prints RESULT.md (table first)."""
import sys, os, json, re, statistics as st
D = sys.argv[1]
def rd(p):
    try: return [json.loads(l) for l in open(p)]
    except OSError: return []
ev = rd(D + '/turns.jsonl'); meta = json.load(open(D + '/soak.json')) if os.path.exists(D + '/soak.json') else {}
turns = [e for e in ev if e.get('kind') == 'turn']; ok = [e for e in turns if not e.get('bad')]
chk = [e for e in ev if e.get('kind') == 'restore_check']; can = [e for e in ev if e.get('kind') == 'cancel']; died = [e for e in ev if e.get('kind') == 'container_died']
vr = []
try:
    for l in open(D + '/vram.csv').read().splitlines()[1:]:
        p = l.split(','); vr.append((float(p[0]), int(float(p[1])), int(p[3]), float(p[4]) if p[4] not in ('', '[N/A]') else 0.0))
except OSError: pass
srv = open(D + '/server.log', errors='replace').read() if os.path.exists(D + '/server.log') else ''
fatal = [l for l in srv.splitlines() if re.search(r'CUDA error|out of memory|cudaMalloc failed|GGML_ASSERT|segmentation|Aborted|failed to (decode|allocate)|decode: failed|llama_decode.*(fail|error)', l, re.I)]
errl = open(D + '/errors.log').read().splitlines() if os.path.exists(D + '/errors.log') else []
httpbad = [e for e in turns if e.get('http') != 200]; recbad = [e for e in turns if e.get('http') == 200 and not e.get('receipt_ok')]
nerr = len(httpbad) + len(recbad) + len(died) + len(fatal) + (1 if meta.get('error') else 0)
ident = sum(1 for c in chk if c.get('identical')); modes = sorted({c.get('mode') for c in chk})
def med(x): return st.median(x) if x else None
def drift(sel, key, kind=None):
    s = sorted([e for e in ok if sel(e) and e.get(key)], key=lambda e: e['t_send'])
    if len(s) < 6: return None
    n = len(s) // 3; a = med([e[key] for e in s[:n]]); b = med([e[key] for e in s[-n:]]); return (a, b, (b / a - 1) * 100 if a else None, len(s))
dec = lambda e: e.get('phase') == 'turn'
D1 = drift(lambda e: dec(e) and e.get('size') == '1K', 'predicted_tps'); D4 = drift(lambda e: dec(e) and e.get('size') == '4K', 'predicted_tps')
DP = drift(lambda e: dec(e) and e.get('first'), 'prompt_tps'); DA = drift(lambda e: dec(e) and not e.get('first'), 'prompt_tps')
def fmt(d): return 'n/a (too few)' if not d else '%.2f -> %.2f (%+.1f%%, n=%d)' % (d[0], d[1], d[2] or 0, d[3])
free_min = min((v[2] for v in vr), default=None); used_max = max((v[1] for v in vr), default=None)
half = len(vr) // 2
fm_first = min((v[2] for v in vr[:half]), default=None); fm_last = min((v[2] for v in vr[half:]), default=None)
wall = (meta.get('deadline', 0) - meta.get('t0', 0)) / 60 if meta.get('t0') else None
verdict = 'PASS' if (nerr == 0 and chk and ident == len(chk) and vr and meta.get('gemm_check', {}).get('ok')) else 'FAIL/REVIEW'
print('# S0 service soak result (batch8 prod build, 2 slots ~96K)\n')
print('| metric | value |\n|---|---|')
print('| verdict | %s |' % verdict)
print('| soak length (min, configured) | %s |' % (('%.0f' % wall) if wall else 'n/a'))
print('| turns ok / total | %d / %d |' % (len(ok), len(turns)))
print('| errors (HTTP != 200, bad receipt, container death, fatal server-log lines, driver error) | %d (%d, %d, %d, %d, %d) |' % (nerr, len(httpbad), len(recbad), len(died), len(fatal), 1 if meta.get('error') else 0))
print('| token identity after restore (greedy 64) | %d / %d identical (mode %s) |' % (ident, len(chk), ','.join(m or '?' for m in modes) or 'n/a'))
print('| mid-prefill cancels honoured (slot idle < 120 s) / total | %d / %d; idle-after median %s s |' % (sum(1 for c in can if c.get('honoured')), len(can), ('%.1f' % med([c['idle_after_s'] for c in can if c.get('idle_after_s') is not None])) if any(c.get('idle_after_s') is not None for c in can) else 'n/a'))
print('| peak VRAM used (MiB) | %s |' % used_max)
print('| min free VRAM (MiB) | %s (first half %s, second half %s) |' % (free_min, fm_first, fm_last))
print('| decode tok/s 1K turns, first third -> last third | %s |' % fmt(D1))
print('| decode tok/s 4K turns, first third -> last third | %s |' % fmt(D4))
print('| prefill tok/s (fresh ~84K turn 0) first -> last third | %s |' % fmt(DP))
print('| prefill tok/s (append turns) first -> last third | %s |' % fmt(DA))
print('| GEMM engaged (server.log) | %s |' % meta.get('gemm_check', {}).get('engaged', 'n/a'))
print('| mean GPU power (W) / max temp | %s / %s |' % (('%.0f' % st.mean([v[3] for v in vr])) if vr else 'n/a', 'see vram.csv'))
print('\n## Method\nOne fresh container ns0-* from the L0 prod arm (libs prod-libs/batch8-20261115, env of run-35b-batch8-20261115.sh, cap 1024, ub 2048, b 2048, 2 slots, ctx 196608, PREFILL_SINGLE_STREAM, cache-ram 0, ckpt 16/2048). '
      'Each slot loops episodes: fresh ~84K base prefill, then 4 N3-short tool-append turns (1K/4K) to ~96K. Every episode: after turn 1 the other slot is drained, slot saved via the server API, greedy 64 tokens (pre), erase, restore, greedy 64 tokens (post), compare token ids. '
      'Odd episodes start with a streaming request cancelled 20-50 s into prefill, then time to slot idle is logged. VRAM sampled every 5 s (vram.csv). Per-turn timings in turns.jsonl.')
if errl: print('\n## errors.log (first 15)\n```\n' + '\n'.join(errl[:15]) + '\n```')
if fatal: print('\n## server.log fatal-pattern lines (first 10)\n```\n' + '\n'.join(fatal[:10]) + '\n```')
bad = [c for c in chk if not c.get('identical')]
if bad: print('\n## identity failures\n' + '\n'.join('- %s mode=%s first_diff=%s n_pre=%s n_post=%s restore_http=%s' % (c['tag'], c.get('mode'), c.get('first_diff'), c.get('n_pre'), c.get('n_post'), c.get('restore_http')) for c in bad))
print('\n## Files\nturns.jsonl, vram.csv, errors.log, soak.json, server.log, libs.sha256, steps.log in the run dir.')
