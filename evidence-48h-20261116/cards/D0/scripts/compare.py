#!/usr/bin/env python3
"""Per-slot token identity between N3 replay sessions.
usage: compare.py <session> <session> ...   (session = a dir holding runs/*/run.json, or a run.json path; first = reference)
Reads reqs[].gen_ids (greedy generated token ids per agent/slot turn, saved by n3_replay_d0.py / n3_replay_a5.py).
Reports per slot (agent) and turn: identical / first divergent position, and for each other run vs the reference the first divergence
(agent, turn, position). Later turns after a divergence are not comparable (prompts differ), so only the first divergence is meaningful.
Also usable as a module: load(path) -> {(agent, turn): ids}, first_div(ref, other) -> None | (agent, turn, pos, ref_tok, other_tok, note)."""
import sys, json, glob, os
def run_files(p):
    if p.endswith('.json'): return [p]
    return sorted(glob.glob(p + '/runs/*/run.json')) or sorted(glob.glob(p + '/run.json'))
def load(p):
    out = {}
    for f in run_files(p):
        R = json.load(open(f))
        if R.get('kind') != 'main': continue
        for r in R.get('reqs', []):
            if 'gen_ids' in r: out[(r['agent'], r['turn'])] = r['gen_ids']
    return out
def first_div(ref, oth):
    for key in sorted(set(ref) | set(oth), key=lambda k: (k[1], k[0])):   # turn-major: earliest turn first
        a, b = ref.get(key), oth.get(key)
        if a is None or b is None: return (key[0], key[1], 0, None, None, 'missing turn in %s' % ('ref' if a is None else 'other'))
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]: return (key[0], key[1], i, a[i], b[i], 'token differs')
        if len(a) != len(b): return (key[0], key[1], min(len(a), len(b)), None, None, 'length differs %d vs %d' % (len(a), len(b)))
    return None
def slot_summary(ref, oth):
    """per agent: first diverging turn/pos or 'identical (n turns, n tokens)'"""
    res = {}
    for ag in sorted({k[0] for k in ref} | {k[0] for k in oth}):
        keys = sorted((k for k in set(ref) | set(oth) if k[0] == ag), key=lambda k: k[1]); d = None; ntok = 0
        for k in keys:
            a, b = ref.get(k), oth.get(k)
            if a is None or b is None: d = (k[1], 0, 'missing'); break
            m = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
            if m is None and len(a) != len(b): m = min(len(a), len(b))
            if m is not None: d = (k[1], m, 'differs'); break
            ntok += len(a)
        res[ag] = ('identical (%d turns, %d tokens)' % (len(keys), ntok)) if d is None else ('FIRST DIVERGENCE turn %d pos %d (%s)' % d)
    return res
def main():
    paths = sys.argv[1:]
    if len(paths) < 2: sys.exit(__doc__)
    runs = [load(p) for p in paths]
    for p, r in zip(paths, runs):
        if not r: print('WARNING: no gen_ids in', p)
    print('| run | slot 0 vs ref | slot 1 vs ref |\n|---|---|---|')
    for i, (p, r) in enumerate(zip(paths, runs)):
        if i == 0: print('| 0 (ref) %s | - | - |' % os.path.basename(p.rstrip('/'))); continue
        s = slot_summary(runs[0], r); print('| %d %s | %s | %s |' % (i, os.path.basename(p.rstrip('/')), s.get(0, 'n/a'), s.get(1, 'n/a')))
    ident = sum(1 for r in runs[1:] if first_div(runs[0], r) is None)
    print('\nruns token-identical to reference (all slots, all turns): %d / %d' % (ident, len(runs) - 1))
    print('deterministic run to run: %s' % ('YES' if ident == len(runs) - 1 and runs[0] else 'NO'))
if __name__ == '__main__': main()
