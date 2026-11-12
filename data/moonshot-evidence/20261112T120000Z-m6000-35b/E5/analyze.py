#!/usr/bin/env python3
"""E5 analysis. usage: analyze.py --gguf MODEL.gguf [--decode P.decode.csv] [--prefill P.prefill.csv]
Run with: python3 -I analyze.py ... (gguf-py is added from src/ via path below)."""
import argparse, csv, os, sys, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src", "gguf-py"))

def expert_bytes(path):
    """bytes per expert per layer (gate+up+down exps) from tensor shapes/types."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    per = collections.defaultdict(int)
    for t in r.tensors:
        n = t.name
        if n.startswith("blk.") and "_exps.weight" in n and "ffn_" in n:
            il = int(n.split(".")[1]); ne = int(t.shape[-1])  # shape [.., .., n_expert]
            per[il] += int(t.n_bytes) // ne
    return dict(per)

def gini(v):
    v = sorted(v); n = len(v); s = sum(v)
    if n == 0 or s == 0: return 0.0
    return (2 * sum((i + 1) * x for i, x in enumerate(v)) / (n * s)) - (n + 1) / n

def decode(csvp, eb):
    d = collections.defaultdict(lambda: collections.defaultdict(set))  # (step,layer)->seq->set
    for row in csv.reader(open(csvp)):
        st, il, sq, rk, e = map(int, row)
        d[(st, il)][sq].add(e)
    layers = sorted({k[1] for k in d}); steps = sorted({k[0] for k in d})
    print(f"decode: {len(steps)} steps, {len(layers)} MoE layers")
    jac = collections.defaultdict(list); reuse = collections.defaultdict(list)
    n1 = n2 = 0; per_step = []
    for st in steps:
        s1 = s2 = 0
        for il in layers:
            a = d[(st, il)].get(0, set()); b = d[(st, il)].get(1, set())
            if a or b: jac[il].append(len(a & b) / max(1, len(a | b)))
            for q, cur in ((0, a), (1, b)):
                prev = d.get((st - 1, il), {}).get(q)
                if prev is not None and cur: reuse[(il, q)].append(len(prev & cur) / len(cur))
            w = eb.get(il, 0)
            s1 += (len(a) + len(b)) * w; s2 += len(a | b) * w   # N1: each seq separately (2 single-token steps)
        per_step.append((s1, s2)); n1 += s1; n2 += s2
    print("layer  jaccard(seq0,seq1)  reuse_seq0  reuse_seq1")
    for il in layers:
        m = lambda x: f"{statistics.mean(x):.3f}" if x else "  -  "
        print(f"{il:5d}  {m(jac[il])}              {m(reuse[(il,0)])}       {m(reuse[(il,1)])}")
    allj = [x for v in jac.values() for x in v]
    print(f"MEAN jaccard {statistics.mean(allj):.3f}")
    for q in (0, 1):
        r = [x for (il, qq), v in reuse.items() if qq == q for x in v]
        if r: print(f"MEAN adjacent-step reuse seq{q}: {statistics.mean(r):.3f}")
    if n1:
        print(f"unique expert weight bytes/step: N1 (2 seqs separately) {n1/len(steps)/2**20:.1f} MiB/step-pair, "
              f"N2 (union) {n2/len(steps)/2**20:.1f} MiB/step, N2/N1 = {n2/n1:.3f}")

def prefill(csvp):
    sk = []; gi = []; per_layer = collections.defaultdict(list)
    for row in csv.reader(open(csvp)):
        ub, il, nt = int(row[0]), int(row[1]), int(row[2]); c = list(map(int, row[3:]))
        m = sum(c) / len(c)
        if m: sk.append(max(c) / m); gi.append(gini(c)); per_layer[il].append(max(c) / m)
    print(f"prefill: {len(sk)} (ubatch,layer) histograms")
    print(f"max/mean rows per expert: mean {statistics.mean(sk):.2f} median {statistics.median(sk):.2f} max {max(sk):.2f}")
    print(f"Gini: mean {statistics.mean(gi):.3f} max {max(gi):.3f}")
    for il in sorted(per_layer): print(f"  layer {il:3d} max/mean {statistics.mean(per_layer[il]):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gguf"); ap.add_argument("--decode"); ap.add_argument("--prefill")
    a = ap.parse_args()
    if a.decode:
        eb = expert_bytes(a.gguf) if a.gguf else {}
        decode(a.decode, eb)
    if a.prefill: prefill(a.prefill)
