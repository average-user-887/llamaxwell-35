#!/usr/bin/env python3
# L2 summarizer: python3 -I summarize.py <run dir>. Reads T_*/nt/*.nt.tsv (last file per run = final dump).
import sys, glob, collections
PEAK_F, PEAK_B = 6.8e12, 317e9
N3 = {'prefill': 40.0, 'mixed': 60.0, 'decode': 49.0}; N3W = 149.0   # s, from L0 wall split
def cls(op, w):
    if op.startswith('FLASH_ATTN'): return 'flash-attn'
    if op.startswith('MUL_MAT_ID'):
        for k in ('gate_up', 'gate', 'up', 'down'):
            if k in w: return 'MoE ' + k
        return 'MoE other'
    if op.startswith('GATED_DELTA') or op.startswith('SSM_CONV') or op == 'SSM_SCAN': return 'GDN'
    if op.startswith('MUL_MAT'): return 'dense mmq'
    if op in ('RMS_NORM', 'NORM', 'L2_NORM', 'ROPE', 'ROPE_BACK'): return 'norm/rope'
    return 'other'
rd = sys.argv[1]; files = []
for d in sorted(glob.glob(rd + '/T_*/nt')):
    f = sorted(glob.glob(d + '/*.nt.tsv'))
    files += f
agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0]); phase_ms = collections.defaultdict(float); gc = collections.Counter()
for f in files:
    for ln in open(f):
        p = ln.rstrip('\n').split('\t')
        if ln.startswith('#graphs'):
            for kv in p[1:]: k, v = kv.split('='); gc[k] += int(v)
            continue
        if ln.startswith('#') or len(p) < 6: continue
        ph, opw, calls, ms, fl, by = p[0], p[1], int(p[2]), float(p[3]), float(p[4]), float(p[5])
        op, _, w = opw.partition(':'); c = cls(op, w)
        a = agg[(ph, c)]; a[0] += ms; a[1] += fl; a[2] += by; a[3] += calls; phase_ms[ph] += ms
print('graphs:', dict(gc), ' timed ms by phase:', {k: round(v) for k, v in phase_ms.items()})
print('\n| phase | class | ms | %% of phase | %% of N3 wall (phase share x N3 phase wall / %.0f s) | achieved TFLOPS | %% fp32 peak | achieved GB/s | %% BW peak | roofline bound |' % N3W)
print('|---|---|---|---|---|---|---|---|---|---|')
wall = {'prefill': N3['prefill'], 'small': N3['mixed'], 'decode': N3['decode']}
for ph in ('prefill', 'small', 'decode'):
    tot = phase_ms.get(ph, 0) or 1
    for (p2, c), a in sorted(((k, v) for k, v in agg.items() if k[0] == ph), key=lambda kv: -kv[1][0]):
        s = a[0] / 1e3; tf = a[1] / s / 1e12 if s else 0; bw = a[2] / s / 1e9 if s else 0
        t_f, t_b = a[1] / PEAK_F, a[2] / PEAK_B
        print('| %s | %s | %.0f | %.1f | %.1f | %.2f | %.0f | %.0f | %.0f | %s |' % (ph, c, a[0], 100 * a[0] / tot, 100 * (a[0] / tot) * wall[ph] / N3W, tf, 100 * tf * 1e12 / PEAK_F, bw, 100 * bw * 1e9 / PEAK_B, 'compute' if t_f >= t_b else 'memory'))
print('\nNote: phase wall assumed: prefill-only 40 s, mixed 60 s (graphs with 2..1023 rows), decode 49 s; shares are of GPU node time (sync-per-graph timing), not host wall. Roofline: fp32 6.8 TFLOPS, 317 GB/s; mmq int8 on sm_52 has no faster path, so fp32 peak is the ceiling used.')
