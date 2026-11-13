#!/usr/bin/env python3
"""NC2 read-only inventory of raw logit vectors (G1 .npy, G1c .f32). usage: python3 -I inventory.py"""
import os, re, json, glob, hashlib, struct
M = '<LAB>'; E = M + '/occamy35-day-20261112/evidence'
G1 = E + '/G1/20261112T171109Z'; G1C = E + '/G1c/run-20261112T202254Z'; OUT = os.path.dirname(os.path.abspath(__file__))
NV = 248320
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''): h.update(c)
    return h.hexdigest()
def npy_shape(p):
    with open(p, 'rb') as f:
        assert f.read(6) == b'\x93NUMPY'; maj, _ = f.read(2); n = struct.unpack('<H' if maj == 1 else '<I', f.read(2 if maj == 1 else 4))[0]
        hd = eval(f.read(n).decode().strip()); return hd['shape'], hd['descr']
rec = []
ARMNAME = {'A': 'A', 'Ap': 'Ap', 'B': 'B', 'C': 'C(F32 ref)', 'A2': 'A2(repeat)', 'B2': 'B2(repeat)'}
for p in sorted(glob.glob(G1 + '/*/*.npy')):
    arm = p.split('/')[-2]; m = re.match(r'(.+)_r(\d+)_(\d+)\.npy', os.path.basename(p)); shp, dt = npy_shape(p)
    rec.append(dict(set='G1', arm=arm, arm_label=ARMNAME[arm], bucket=int(m.group(3)), doc=m.group(1), rep=int(m.group(2)), rows=1, dims=list(shp), dtype=dt, file=p.replace(M + '/', '<M>/'), bytes=os.path.getsize(p), sha256=sha(p)))
for p in sorted(glob.glob(G1C + '/*/doc*_b*.f32')):
    arm = p.split('/')[-2]; m = re.match(r'doc(\d+)_b(\d+)\.f32', os.path.basename(p)); sz = os.path.getsize(p); rows = sz // (4 * NV)
    rec.append(dict(set='G1c', arm=arm, arm_label=ARMNAME[arm], bucket=int(m.group(2)), doc='doc' + m.group(1), rep=0, rows=rows, dims=[rows, NV], dtype='<f4', file=p.replace(M + '/', '<M>/'), bytes=sz, sha256=sha(p), size_ok=(sz == rows * NV * 4)))
# expected vs present
miss = []
have = {(r['set'], r['arm'], r['doc'], r['bucket']) for r in rec}
for arm in ('A', 'B'):
    for k in range(8):
        for B in (32768, 65536, 98304):
            if ('G1c', arm, 'doc%d' % k, B) not in have: miss.append(('G1c', arm, 'doc%d' % k, B, 'expected, absent'))
for arm, why in (('Ap', 'by design: noise arm ran doc 0 only (tier 2)'), ('C', 'by design: F32 ref ran doc 0 only (tier 2)')):
    for k in range(8):
        for B in (32768, 65536, 98304):
            if ('G1c', arm, 'doc%d' % k, B) not in have: miss.append(('G1c', arm, 'doc%d' % k, B, why + ('; 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only)' if arm == 'C' and B == 98304 else '')))
g1docs = sorted({r['doc'] for r in rec if r['set'] == 'G1'}); cks = sorted({r['bucket'] for r in rec if r['set'] == 'G1'})
for arm in ('A', 'Ap', 'B', 'C'):
    for d in g1docs:
        for c in (16384, 32768, 65536, 65836):
            if d == 'codexs' and c == 65536 or d != 'codexs' and c == 65836: continue
            if ('G1', arm, d, c) not in have: miss.append(('G1', arm, d, c, 'absent (arm did not run this doc/checkpoint)'))
tot = sum(r['bytes'] for r in rec); big = [r for r in rec if r['bytes'] > 50 * 10**6]
by = {}
for r in rec: by.setdefault(r['set'], [0, 0]); by[r['set']][0] += 1; by[r['set']][1] += r['bytes']
summary = dict(files=len(rec), total_bytes=tot, by_set=by, files_over_50MB=len(big), nv_expected=NV,
               git_push_fit=dict(each_file_lt_50MB=(len(big) == 0), total_lt_300MB=(tot < 300 * 10**6), G1_alone_total_bytes=by['G1'][1], G1c_alone_total_bytes=by['G1c'][1], note='MB = 10^6 bytes'))
json.dump(dict(summary=summary, files=rec, missing=[dict(set=a, arm=b, doc=c, bucket=d, reason=e) for a, b, c, d, e in miss]), open(OUT + '/manifest.json', 'w'), indent=1)
L = ['# NC2 logit-vector manifest', '', 'Generated read-only from existing files; nothing re-run. `<M>` = the m6000-llama-cpp root. Full per-file list with sha256: manifest.json.', '',
     '## Totals', '', '| set | files | bytes | MB |', '|---|---|---|---|']
for s, (n, b) in by.items(): L.append('| %s | %d | %d | %.1f |' % (s, n, b, b / 1e6))
L += ['| all | %d | %d | %.1f |' % (len(rec), tot, tot / 1e6), '', 'Git push fit (each file <50 MB, total <300 MB): **%s**. G1c files are %.1f MB each (64 rows x %d x 4 B): %d of %d files exceed 50 MB; total %.0f MB vs 300 MB. G1 .npy files (%.1f KB each, 1 row) alone fit: %.2f MB.' % (
     'NO' if big or tot >= 300e6 else 'yes', 63569920 / 1e6, NV, len(big), len(rec), tot / 1e6, 248320 * 4 / 1e3, by['G1'][1] / 1e6), '', '## G1c (rows = teacher-forced decode rows [64 x %d f32] at positions B..B+63; run-20261112T202254Z)' % NV, '',
     '| arm | bucket | docs present | rows each | dims | bytes each |', '|---|---|---|---|---|---|']
for arm in ('A', 'Ap', 'B', 'C'):
    for B in (32768, 65536, 98304):
        x = [r for r in rec if r['set'] == 'G1c' and r['arm'] == arm and r['bucket'] == B]
        if x: L.append('| %s | %d | %s | %s | %s | %s |' % (ARMNAME[arm], B, ','.join(r['doc'][3:] for r in x), ','.join(sorted({str(r['rows']) for r in x})), ','.join(sorted({str(r['dims']) for r in x})), ','.join(sorted({str(r['bytes']) for r in x}))))
        else: L.append('| %s | %d | none | - | - | - |' % (ARMNAME[arm], B))
L += ['', '## G1 (1 row per file: single position, `<doc>_r<rep>_<checkpoint>.npy`, dims (%d,) float32, %d bytes + header; run 20261112T171109Z)' % (NV, 248320 * 4), '', '| arm | docs | checkpoints (file count) |', '|---|---|---|']
for arm in ('A', 'Ap', 'B', 'C', 'A2', 'B2'):
    x = [r for r in rec if r['set'] == 'G1' and r['arm'] == arm]
    L.append('| %s | %s | %s (%d files) |' % (ARMNAME[arm], ','.join(sorted({r['doc'] for r in x})), ','.join(str(c) for c in sorted({r['bucket'] for r in x})), len(x)))
L += ['', 'Earlier G1 attempts (20261112T170708Z, T170929Z, T170954Z) hold no logit vectors (only logs/rows.csv).', '', '## Missing', '', '| set | arm | doc | bucket/checkpoint | reason |', '|---|---|---|---|---|']
for a, b, c, d, e in miss: L.append('| %s | %s | %s | %d | %s |' % (a, b, c, d, e))
open(OUT + '/MANIFEST.md', 'w').write('\n'.join(L) + '\n')
print(json.dumps(summary, indent=1)); print(len(miss), 'missing rows')
