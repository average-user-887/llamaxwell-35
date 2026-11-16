#!/usr/bin/env python3
# DC0 summarizer: python3 -I dc0_summarize.py <run dir> [file.tsv]. Per class: 1-row (decode) vs 2-row (dec2) ms/step, nodes/step, us/node, ratio.
import sys, glob, collections
def cls(op):
    o, _, w = op.partition(':'); tag = ''
    if '@' in o: o, tag = o.split('@')
    if o.startswith('FLASH_ATTN'): return 'attention'
    if o.startswith('MUL_MAT_ID'): return 'MoE experts'
    if o in ('GATED_DELTA_NET', 'SSM_CONV'): return 'GDN'
    if o.startswith('MUL_MAT'): return 'dense mmq'
    if o in ('RMS_NORM', 'NORM', 'L2_NORM', 'ROPE'):
        return {'MoE': 'norm-led chain fused w/ MoE', 'GDN': 'norm-led chain fused w/ GDN', 'FA': 'norm-led chain fused w/ FA', 'fused': 'norm-led chain fused other'}.get(tag, 'norm/rope')
    return 'other'
rd = sys.argv[1]; files = [sys.argv[2]] if len(sys.argv) > 2 else [sorted(glob.glob(d + '/*.nt.tsv'))[-1] for d in sorted(glob.glob(rd + '/T_*/nt')) if glob.glob(d + '/*.nt.tsv')]
G = collections.Counter(); GM = collections.Counter(); A = collections.defaultdict(lambda: [0.0, 0, 0])
for f in files:
    for ln in open(f):
        p = ln.rstrip('\n').split('\t')
        if ln.startswith('#graphs'):
            for kv in p[1:]: k, v = kv.split('='); G[k] += int(v)
        elif ln.startswith('#graphms'):
            for kv in p[1:]: k, v = kv.split('='); GM[k] += float(v)
        elif not ln.startswith('#') and len(p) >= 6:
            a = A[(p[0], cls(p[1]))]; a[0] += float(p[3]); a[1] += int(p[2]); a[2] += int(p[6]) if len(p) > 6 else 0
print('graphs:', dict(G), ' graph ms:', {k: round(v) for k, v in GM.items()})
n1, n2 = G.get('decode', 0) or 1, G.get('dec2', 0) or 1
print('\n| class | 1-row ms/step | 1-row nodes/step | 1-row us/node | 2-row ms/step | 2-row nodes/step | 2-row us/node | ratio 2/1 | flag |\n|---|---|---|---|---|---|---|---|---|')
for c in sorted({k[1] for k in A}, key=lambda c: -A[('decode', c)][0]):
    d, e = A.get(('decode', c), [0, 0, 0]), A.get(('dec2', c), [0, 0, 0])
    m1, m2 = d[0] / n1, e[0] / n2; u1 = 1e3 * d[0] / d[1] if d[1] else 0; u2 = 1e3 * e[0] / e[1] if e[1] else 0
    r = m2 / m1 if m1 else 0; fl = []
    if c in ('MoE experts', 'dense mmq', 'GDN') or c.startswith('norm-led chain fused w/ MoE') and r > 1.3: pass
    if r > 1.3 and c not in ('attention', 'norm/rope', 'other'): fl.append('ratio>1.3')
    if 8 <= u2 <= 12 or 8 <= u1 <= 12: fl.append('~launch-bound(8-12us/node)')
    print('| %s | %.2f | %.1f | %.1f | %.2f | %.1f | %.1f | %.2f | %s |' % (c, m1, d[1] / n1, u1, m2, e[1] / n2, u2, r, ' '.join(fl)))
t1 = GM.get('decode', 0) / n1; t2 = GM.get('dec2', 0) / n2
print('\nstep total ms (timed graph): 1-row %.2f, 2-row %.2f, ratio %.2f' % (t1, t2, t2 / t1 if t1 else 0))
