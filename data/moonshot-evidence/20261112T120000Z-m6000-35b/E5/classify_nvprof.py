#!/usr/bin/env python3
"""Kernel-family classifier for nvprof --print-gpu-summary logs.
usage: classify_nvprof.py BASE.txt K1.txt K2.txt K3.txt [--calls N]   (N = warm+calls timed calls per k run, default 23)
Per-call = (log_k - BASE) / N ; BASE is the --only-k 0 run (load+gen+restore, no timed calls)."""
import re, sys, collections
FAM = [("attention/FA", r"fattn|flash_attn|occamy_fa|ofa_|soft_max|rope|kv|cpy.*(k|v)cache|set_rows"),
       ("GDN/ssm", r"gdn|gated_delta|ssm_conv|ssm_scan|delta_net|l2_norm"),
       ("MoE", r"moe|mul_mat_id|mmid|topk|argsort|expert|down[xh]|mmq_ids|quantize_mmq|glu|swiglu|sum_rows|k_get_rows"),
       ("lm head", r"head|rowpack|output"),
       ]
def fam(n):
    l = n.lower()
    for f, rx in FAM:
        if re.search(rx, l): return f
    return "other"
def parse(p):
    d = {}
    for ln in open(p, errors="replace"):
        m = re.match(r"\s*(?:GPU activities:)?\s*([\d.]+)%\s+([\d.]+)(ns|us|ms|s)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$", ln)
        if not m: continue
        t = float(m.group(2)) * {"ns": 1e-6, "us": 1e-3, "ms": 1, "s": 1e3}[m.group(3)]
        n = m.group(5).strip()
        if n.startswith("[CUDA"): continue
        d[n] = (t, int(m.group(4)))
    return d
a = sys.argv[1:]; N = 23
if "--calls" in a: i = a.index("--calls"); N = int(a[i + 1]); del a[i:i + 2]
base = parse(a[0]); res = {}
for k, p in zip((1, 2, 3), a[1:4]):
    d = parse(p); agg = collections.defaultdict(lambda: [0.0, 0.0])
    for n, (t, c) in d.items():
        bt, bc = base.get(n, (0.0, 0))
        dt, dc = (t - bt) / N, (c - bc) / N
        if dt <= 0 and dc <= 0: continue
        agg[fam(n)][0] += dt; agg[fam(n)][1] += dc
    res[k] = agg
print("%-14s" % "family" + "".join("  k=%d ms/call  calls" % k for k in (1, 2, 3)))
for f in [x[0] for x in FAM] + ["other"]:
    print("%-14s" % f + "".join("  %11.3f %7.1f" % tuple(res[k][f]) for k in (1, 2, 3)))
print("%-14s" % "TOTAL" + "".join("  %11.3f %7.1f" % (sum(v[0] for v in res[k].values()), sum(v[1] for v in res[k].values())) for k in (1, 2, 3)))
