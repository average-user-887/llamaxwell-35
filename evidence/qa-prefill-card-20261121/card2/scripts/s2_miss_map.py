"""Map forced-full-reprocess events in a BFCL server.log to sample ids (k-th 'new prompt' task == k-th cache-requests row),
and count them per frozen-stage prefix (smoke/early/preview/full)."""
import json, re, sys, collections
log, jl, freeze = sys.argv[1:4]
rows = [json.loads(l) for l in open(jl)]
rx_new = re.compile(r'task (\d+) \| new prompt, .*task.n_tokens = (\d+)')
rx_pe = re.compile(r'task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens')
tasks = []; idx = {}; cur = None
for ln in open(log, errors='replace'):
    m = rx_new.search(ln)
    if m: idx[int(m[1])] = len(tasks); tasks.append(dict(n=int(m[2]), full=False)); cur = len(tasks) - 1; continue
    if 'forcing full prompt re-processing' in ln and cur is not None: tasks[cur]['full'] = True
    m = rx_pe.search(ln)
    if m and int(m[1]) in idx: tasks[idx[int(m[1])]].update(ms=float(m[2]), tok=int(m[3]))
print('tasks', len(tasks), 'rows', len(rows))
assert len(tasks) == len(rows)
per = collections.defaultdict(lambda: [0, 0.0, 0])
for t, r in zip(tasks, rows):
    p = per[r['id']]; p[2] += 1
    if t['full'] and r['request'] > 1: p[0] += 1; p[1] += t.get('ms', 0) / 1e3
fz = json.load(open(freeze)); cohort = [c['id'] if isinstance(c, dict) else c for c in fz['cohort']]
for name, n in (('early', 16), ('preview', 40), ('full', 160)):
    ids = cohort[:n]; print(name, n, 'misses', sum(per[i][0] for i in ids), 'miss_s %.1f' % sum(per[i][1] for i in ids), 'requests', sum(per[i][2] for i in ids))
hit = sorted(((v[0], round(v[1], 1), k, cohort.index(k) if k in cohort else -1) for k, v in per.items() if v[0]), reverse=True)
print('samples with misses', len(hit)); print(hit[:30])
