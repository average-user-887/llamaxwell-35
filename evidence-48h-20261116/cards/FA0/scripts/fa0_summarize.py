#!/usr/bin/env python3
# python3 -I fa0_summarize.py <run dir>: per config, per-phase ms/op for ops with keys>=81920, speedup vs mean of A and A2.
import sys, glob, os
rd = sys.argv[1]; rows = {}
for d in sorted(glob.glob(rd + '/T_*')):
    n = os.path.basename(d)[2:]; f = d + '/nt/fa0.tsv'
    if not os.path.exists(f): rows[n] = None; continue
    r = {}
    for ln in open(f):
        p = ln.rstrip('\n').split('\t')
        if ln.startswith('#') or p[0] != 'keys_ge_81920': continue
        r[p[3]] = (int(p[1]), float(p[4]))
    rows[n] = r
PH = ['pack', 'convert_kv', 'qk_sgemm', 'softmax_mask', 'pv_sgemm', 'reduce', 'op_total']
def per(r, k): ops = r['op_total'][0] or 1; return r[k][1] / ops
base = [rows[k] for k in ('A', 'A2') if rows.get(k)]
if not base: print('no baseline data'); sys.exit(1)
b = {k: sum(per(x, k) for x in base) / len(base) for k in PH}
noise = abs(per(base[0], 'op_total') - per(base[-1], 'op_total')) / b['op_total'] if len(base) > 1 else float('nan')
print('baseline ms/op (keys>=81920): ' + ', '.join('%s %.2f' % (k, b[k]) for k in PH))
gem = b['qk_sgemm'] + b['pv_sgemm']; non = b['op_total'] - gem
print('non-GEMM share of FA op time (pack+convert+softmax+reduce+gaps): %.1f%%   (GEMM-only %.1f%%, A vs A2 noise %.1f%%)' % (100 * non / b['op_total'], 100 * gem / b['op_total'], 100 * noise))
print('\n| config | ops | op ms | speedup vs A | ' + ' | '.join(PH[:-1]) + ' (ms/op) |\n|---|---|---|---|' + '---|' * (len(PH) - 1))
for n, r in rows.items():
    if not r: print('| %s | no data |' % n); continue
    print('| %s | %d | %.2f | %.3fx | %s |' % (n, r['op_total'][0], per(r, 'op_total'), b['op_total'] / per(r, 'op_total'), ' | '.join('%.2f' % per(r, k) for k in PH[:-1])))
