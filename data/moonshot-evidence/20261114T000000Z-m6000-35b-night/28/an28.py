#!/usr/bin/env python3
"""Card 28: per-op split of an nvprof --print-gpu-trace csv of one long prefill.
usage: an28.py trace.csv [--names] ; segments ubatches by a marker kernel (first kernel of each graph: embedding get_rows),
classifies kernels into op classes, prints per-depth-window shares (ms per 2048-token ubatch)."""
import csv, sys, re, collections, json
CLS = [('dense_mm', r'^DENSE:|dequantize_block'), ('moe_mmq', r'mul_mat_q<|mm_ids_helper|mmq_make_tail|quantize_mmq|moe_weighted|topk_moe'), ('attn', r'maxwell_gemm|ofa_|fattn|flash_attn|sgemm|gemm_|gemv|cutlass|magma|cublas|occamy_fa|soft_max'),
       ('moe', r'moe'),
       ('gdn', r'gdn|ssm_conv|conv1d|delta'),
       ('dense_mm', r'mul_mat_q|mmq|mul_mat_vec|mmvq|head_rowpack|quantize_q8|quantize_mmq'),
       ('norm', r'norm'),
       ('rope_kv', r'rope|set_rows|cpy|concat|get_rows')]
def cls(n):
    for c, p in CLS:
        if re.search(p, n, re.I): return c
    return 'other'
def load(p):
    rows = list(csv.reader(open(p))); h = [i for i, r in enumerate(rows) if r and r[0] == 'Start'][0]; hd = rows[h]; ci = {c: i for i, c in enumerate(hd)}
    units = rows[h + 1]; su = 1e-3 if units[ci['Start']] == 'ms' else (1e-6 if units[ci['Start']] == 'us' else 1.0)
    du = {'ms': 1.0, 'us': 1e-3, 's': 1e3, 'ns': 1e-6}[units[ci['Duration']]]
    K = []
    for r in rows[h + 2:]:
        if len(r) != len(hd): continue
        n = r[ci['Name']]
        if n.startswith('['): continue
        n = re.sub(r'\(.*', '', n.replace('void ', ''))[:110]
        K.append((float(r[ci['Start']]) * su, float(r[ci['Duration']]) * du, n, r[ci['Grid X']] + 'x' + r[ci['Grid Y']] + 'x' + r[ci['Grid Z']]))
    K.sort()
    # context: cuBLAS sgemm right after dequantize_block_q4_0 = dense projection (dequant + SGEMM), not attention
    out = []; prev = ''
    for k in K:
        n = k[2]
        if prev.startswith('dequantize_block') and re.search('sgemm', n): n = 'DENSE:' + n
        out.append((k[0], k[1], n, k[3])); prev = n
    return out
K = load(sys.argv[1])
if '--names' in sys.argv:
    c = collections.Counter(); t = collections.Counter()
    for k in K: c[k[2]] += 1; t[k[2]] += k[1]
    for n, v in t.most_common(60): print('%9.1f ms %6d %-8s %s' % (v, c[n], cls(n), n))
    sys.exit()
# ubatch = 30 gated_delta_net_cuda launches (30 GDN layers per graph); kernels attributed by running GDN count
GD = 'gated_delta_net_cuda'; segs = collections.defaultdict(list); g = 0
for k in K:
    segs[g // 30].append(k)
    if k[2].startswith(GD): g += 1
segs = [segs[j] for j in sorted(segs)]
NAMES = []
for c, _ in CLS + [('other', 0)]:
    if c not in NAMES: NAMES.append(c)
out = []
for s in segs:
    busy = sum(k[1] for k in s); wall = (s[-1][0] + s[-1][1] / 1e3 - s[0][0]) * 1e3
    by = collections.Counter(); byk = collections.Counter()
    for k in s: by[cls(k[2])] += k[1]; byk[k[2]] += k[1]
    out.append(dict(n=len(s), busy=busy, wall=wall, by=dict(by), top=byk.most_common(15)))
json.dump(out, open(sys.argv[1] + '.segs.json', 'w'))
for i, o in enumerate(out):
    print('%3d n=%5d busy=%8.1f wall=%8.1f ' % (i, o['n'], o['busy'], o['wall']) + ' '.join('%s=%.0f' % (c, o['by'].get(c, 0)) for c in NAMES))
