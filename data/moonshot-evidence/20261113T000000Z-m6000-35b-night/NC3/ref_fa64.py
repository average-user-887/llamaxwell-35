#!/usr/bin/env python3
"""FP64 CPU reference for GQA flash attention + stage metrics (card N4 / NC3).

Stages (all per (query, head) row):  S=scale*QK^T | S2=S+mask | m=rowmax | l=sum exp(S2-m) | P=exp(S2-m)/l | O=P@V
Shapes follow the dump format in N4_DESIGN.md:
  Q [nq,16,256] f32   K,V [2,nk,256] f16 (compact)   mask [nq_pad,nk] f16   scale f32
  head h uses kv group h//(nh//nkv)   (nh=16, nkv=2 -> h//8)
Usage:  python3 -I ref_fa64.py selftest
        python3 -I ref_fa64.py compare --dir DUMP --layer 19 --gemm GEMM_OUT.f32 --stock STOCK_OUT.f32 [--rows 0,1,63,1023]
Candidate output files: raw f32 [nq,16,256] (full O, as written by the dump driver) and optional
stage files S.f32/S2.f32/P.f32 [nsel_q,16,nk] and ML.f32 [nsel_q,16,2] (m,l) for selected rows.
"""
import json, sys, argparse, os
import numpy as np

# ---------------------------------------------------------------- reference
def ref_rows(Q, K, V, mask, scale, qsel, heads=None, want=("S", "S2", "m", "l", "P", "O")):
    """FP64 reference for selected query rows. Returns dict of float64 arrays [len(qsel),len(heads),...]."""
    nq, nh, D = Q.shape
    nkv, nk, _ = K.shape
    rep = nh // nkv
    heads = list(range(nh)) if heads is None else list(heads)
    out = {k: [] for k in want}
    Kd = K.astype(np.float64); Vd = V.astype(np.float64)
    sc = np.float64(np.float32(scale))          # scale is stored as f32 in op_params
    for qi in qsel:
        rec = {k: [] for k in want}
        for h in heads:
            g = h // rep
            S = (Kd[g] @ Q[qi, h].astype(np.float64)) * sc
            S2 = S + (mask[qi].astype(np.float64) if mask is not None else 0.0)
            m = np.max(S2)
            if not np.isfinite(m):                       # fully masked row -> zeros (GEMM semantics)
                E = np.zeros_like(S2); l = 0.0; P = E; O = np.zeros(D)
            else:
                E = np.exp(S2 - m); l = E.sum(); P = E / l; O = P @ Vd[g]
            for k, v in (("S", S), ("S2", S2), ("m", m), ("l", l), ("P", P), ("O", O)):
                if k in rec: rec[k].append(v)
        for k in want: out[k].append(np.array(rec[k]))
    return {k: np.array(v) for k, v in out.items()}

def ref_rows_naive(Q, K, V, mask, scale, qi, h):
    """Independent scalar-loop implementation (self-test cross-check only; tiny shapes)."""
    nkv = K.shape[0]; rep = Q.shape[1] // nkv; g = h // rep
    nk, D = K.shape[1], K.shape[2]
    s = []
    for j in range(nk):
        a = 0.0
        for d in range(D): a += float(Q[qi, h, d]) * float(K[g, j, d])
        s.append(a * float(np.float32(scale)) + (float(mask[qi, j]) if mask is not None else 0.0))
    m = max(s)
    if m == -np.inf: return np.zeros(D)
    e = [np.exp(x - m) for x in s]; l = sum(e)
    return np.array([sum(e[j] / l * float(V[g, j, d]) for j in range(nk)) for d in range(D)])

# ------------------------------------------------- FP32 emulations (CPU proxies)
def emu_gemm_f32(Q, K, V, mask, scale, qsel, split=4096):
    """Numpy float32 emulation of the v4 GEMM structure: SGEMM scores, online-then-exact softmax,
    P normalised BEFORE PV, PV in `split`-key partial SGEMMs summed ascending. Order inside a BLAS call differs
    from cuBLAS (unknown), so this is a structural proxy, not a bit-match."""
    nq, nh, D = Q.shape; nkv, nk, _ = K.shape; rep = nh // nkv
    O = np.zeros((len(qsel), nh, D), np.float32); ML = np.zeros((len(qsel), nh, 2), np.float32)
    K32 = K.astype(np.float32); V32 = V.astype(np.float32); sc = np.float32(scale)
    for a, qi in enumerate(qsel):
        mrow = mask[qi].astype(np.float32) if mask is not None else np.zeros(nk, np.float32)
        for h in range(nh):
            g = h // rep
            S = (K32[g] @ Q[qi, h].astype(np.float32)) * sc          # alpha applied inside SGEMM; exact for 2^-4
            S2 = S + mrow
            m = np.float32(np.max(S2))
            if not np.isfinite(m): continue
            l = np.float32(np.exp(S2 - m, dtype=np.float32).sum(dtype=np.float32))
            P = (np.exp(S2 - m, dtype=np.float32) * (np.float32(1) / l)).astype(np.float32)
            acc = np.zeros(D, np.float32)
            for s0 in range(0, nk, split):
                acc = acc + P[s0:s0 + split] @ V32[g, s0:s0 + split]
            O[a, h] = acc; ML[a, h] = (m, l)
    return O, ML

def emu_stock_f32(Q, K, V, mask, scale, qsel, tile=256, offset=2.08):
    """Numpy float32 emulation of stock tile math: 256-key tiles, running max (with FATTN_KQ_MAX_OFFSET),
    online rescale of sum and VKQ, single divide at the end. Proxy only."""
    nq, nh, D = Q.shape; nkv, nk, _ = K.shape; rep = nh // nkv
    O = np.zeros((len(qsel), nh, D), np.float32)
    K32 = K.astype(np.float32); V32 = V.astype(np.float32); sc = np.float32(scale)
    for a, qi in enumerate(qsel):
        mrow = mask[qi].astype(np.float32) if mask is not None else np.zeros(nk, np.float32)
        for h in range(nh):
            g = h // rep; q = Q[qi, h].astype(np.float32) * sc       # stock scales Q first
            m = np.float32(-np.inf); l = np.float32(0); acc = np.zeros(D, np.float32)
            for t0 in range(0, nk, tile):
                s = K32[g, t0:t0 + tile] @ q + mrow[t0:t0 + tile]
                mn = np.float32(max(m, np.max(s) + np.float32(offset)))
                if not np.isfinite(mn): continue
                r = np.float32(np.exp(m - mn)) if np.isfinite(m) else np.float32(0)
                p = np.exp(s - mn, dtype=np.float32)
                l = l * r + p.sum(dtype=np.float32); acc = acc * r + p @ V32[g, t0:t0 + tile]; m = mn
            if l > 0: O[a, h] = acc / l
    return O

# ------------------------------------------------------------------ metrics
def ulp_err(test, ref, floor=1e-30):
    """|test-ref| in units of the float32 spacing at ref (ULP-ish). ref float64."""
    ref = np.asarray(ref, np.float64); t = np.asarray(test, np.float64)
    sp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
    sp = np.maximum(sp, floor)
    return np.abs(t - ref) / sp

def stage_metrics(test, ref, name, rel_floor=1e-5, mask_inf=True):
    t = np.asarray(test, np.float64).ravel(); r = np.asarray(ref, np.float64).ravel()
    ok = np.isfinite(r) & np.isfinite(t) if mask_inf else np.ones(r.shape, bool)
    nonfin = int((~np.isfinite(t) & np.isfinite(r)).sum())
    t, r = t[ok], r[ok]
    if r.size == 0: return {"stage": name, "n": 0, "nonfinite_test": nonfin}
    d = np.abs(t - r); u = ulp_err(t, r)
    pc = [50, 90, 99, 99.9]
    return {"stage": name, "n": int(r.size), "nonfinite_test": nonfin, "max_abs": float(d.max()),
            "max_rel": float((d / np.maximum(np.abs(r), rel_floor)).max()), "rmse": float(np.sqrt((d ** 2).mean())),
            "ref_rms": float(np.sqrt((r ** 2).mean())), "ulp_max": float(u.max()),
            **{f"ulp_p{p}": float(np.percentile(u, p)) for p in pc},
            "bit_equal_frac": float((t.astype(np.float32) == r.astype(np.float32)).mean())}

def distance_vs_ref(gemm, stock, ref, name):
    """Who is closer to FP64, element-wise and in RMSE; plus gemm-stock separation."""
    g = np.asarray(gemm, np.float64).ravel(); s = np.asarray(stock, np.float64).ravel(); r = np.asarray(ref, np.float64).ravel()
    eg, es = np.abs(g - r), np.abs(s - r)
    rg, rs = np.sqrt((eg ** 2).mean()), np.sqrt((es ** 2).mean())
    return {"stage": name, "rmse_gemm": float(rg), "rmse_stock": float(rs), "rmse_ratio_gemm_over_stock": float(rg / rs) if rs > 0 else None,
            "frac_gemm_closer": float((eg < es).mean()), "frac_tie": float((eg == es).mean()),
            "max_abs_gemm": float(eg.max()), "max_abs_stock": float(es.max()),
            "rmse_gemm_vs_stock": float(np.sqrt(((g - s) ** 2).mean())),
            "ulp_p99_gemm": float(np.percentile(ulp_err(g, r), 99)), "ulp_p99_stock": float(np.percentile(ulp_err(s, r), 99))}

# ------------------------------------------------------------------ selftest
def _rand_case(rng, nq, nk, D, nh, nkv, causal_off):
    Q = rng.standard_normal((nq, nh, D)).astype(np.float32)
    K = (rng.standard_normal((nkv, nk, D))).astype(np.float16)
    V = (rng.standard_normal((nkv, nk, D))).astype(np.float16)
    mask = np.zeros((nq, nk), np.float16)
    for i in range(nq): mask[i, causal_off + i + 1:] = -np.inf          # causal, tail (padding) masked
    return Q, K, V, mask

def selftest():
    rng = np.random.default_rng(7); fails = 0
    cases = [(5, 70, 16, 4, 2, 40), (3, 300, 32, 4, 2, 250), (4, 129, 8, 16, 2, 100), (2, 64, 16, 2, 1, 10)]
    for (nq, nk, D, nh, nkv, off) in cases:
        Q, K, V, mask = _rand_case(rng, nq, nk, D, nh, nkv, off)
        scale = 1.0 / np.sqrt(D) if D != 16 else 0.0625
        qsel = list(range(nq)); R = ref_rows(Q, K, V, mask, scale, qsel)
        # 1. vs independent naive loop
        for qi in (0, nq - 1):
            for h in (0, nh - 1):
                e = np.abs(R["O"][qi, h] - ref_rows_naive(Q, K, V, mask, scale, qi, h)).max()
                if e > 1e-12: print("FAIL naive", (nq, nk, D), qi, h, e); fails += 1
        # 2. softmax invariants
        if not np.allclose(R["P"].sum(-1), 1.0, atol=1e-12): print("FAIL P sum"); fails += 1
        # 3. FP32 proxies close to fp64 (structure check, split=64 forces multi-split + tail)
        Og, ML = emu_gemm_f32(Q, K, V, mask, scale, qsel, split=64); Os = emu_stock_f32(Q, K, V, mask, scale, qsel, tile=32)
        for nm, O in (("gemm", Og), ("stock", Os)):
            m = stage_metrics(O, R["O"], nm)
            if m["max_abs"] > 2e-5: print("FAIL", nm, m); fails += 1
        if np.abs(ML[..., 0] - R["m"]).max() > 1e-4: print("FAIL m"); fails += 1
        dv = distance_vs_ref(Og, Os, R["O"], "O")
        if not (0 < dv["frac_gemm_closer"] + dv["frac_tie"] <= 1): print("FAIL dist"); fails += 1
    # 4. fully masked row -> zeros, no NaN
    Q, K, V, mask = _rand_case(rng, 2, 40, 8, 4, 2, 10); mask[1, :] = -np.inf
    R = ref_rows(Q, K, V, mask, 0.125, [0, 1])
    if not (np.isfinite(R["O"]).all() and (R["O"][1] == 0).all()): print("FAIL masked row"); fails += 1
    # 5. ulp helper: 1 ulp step at 1.0 is 1.19e-7
    if abs(ulp_err(np.float64(1.0) + 2 ** -23, 1.0) - 1.0) > 1e-9: print("FAIL ulp"); fails += 1
    # 6. no key-padding sensitivity: appending -inf-masked garbage keys must not change O
    Q, K, V, mask = _rand_case(rng, 3, 100, 16, 4, 2, 60)
    Kp = np.concatenate([K, np.full((2, 28, 16), 3.0, np.float16)], 1); Vp = np.concatenate([V, np.full((2, 28, 16), 5.0, np.float16)], 1)
    mp = np.concatenate([mask, np.full((3, 28), -np.inf, np.float16)], 1)
    a = ref_rows(Q, K, V, mask, 0.25, [0, 1, 2], want=("O",))["O"]; b = ref_rows(Q, Kp, Vp, mp, 0.25, [0, 1, 2], want=("O",))["O"]
    if np.abs(a - b).max() > 1e-14: print("FAIL padding invariance"); fails += 1
    print("selftest", "PASS" if fails == 0 else f"FAIL ({fails})"); return fails

# ------------------------------------------------------------------ compare driver
def load(d, name, dtype, shape): return np.fromfile(os.path.join(d, name), dtype).reshape(shape)

def compare(a):
    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    nq, nk, nqp = meta["nq"], meta["nk"], meta["nq_pad"]; L = a.layer
    pre = f"L{L}_"
    Q = load(a.dir, pre + "Q.f32", np.float32, (nq, 16, 256)); K = load(a.dir, pre + "K.f16", np.float16, (2, nk, 256))
    V = load(a.dir, pre + "V.f16", np.float16, (2, nk, 256)); mask = load(a.dir, "mask.f16", np.float16, (nqp, nk))
    rows = [int(x) for x in a.rows.split(",")] if a.rows else sorted({0, 1, nq // 2, nq - 2, nq - 1})
    R = ref_rows(Q, K, V, mask, meta["scale"], rows)
    G = load(a.gemm, "O.f32", np.float32, (nq, 16, 256))[rows] if os.path.isdir(a.gemm) else np.fromfile(a.gemm, np.float32).reshape(nq, 16, 256)[rows]
    S = np.fromfile(a.stock, np.float32).reshape(nq, 16, 256)[rows]
    res = {"meta": meta, "layer": L, "rows": rows, "stages": []}
    for nm in ("S", "S2", "P"):                                  # stage files from the GEMM stage driver (optional)
        f = os.path.join(a.stages or "", f"{pre}{nm}.f32")
        if a.stages and os.path.exists(f):
            T = np.fromfile(f, np.float32).reshape(len(rows), 16, nk); res["stages"].append(stage_metrics(T, R[nm], "gemm_" + nm))
    f = os.path.join(a.stages or "", f"{pre}ML.f32")
    if a.stages and os.path.exists(f):
        ML = np.fromfile(f, np.float32).reshape(len(rows), 16, 2)
        res["stages"] += [stage_metrics(ML[..., 0], R["m"], "gemm_m"), stage_metrics(ML[..., 1], R["l"], "gemm_l")]
    res["stages"] += [stage_metrics(G, R["O"], "gemm_O"), stage_metrics(S, R["O"], "stock_O")]
    res["distance_O"] = distance_vs_ref(G, S, R["O"], "O")
    print(json.dumps(res, indent=1)); return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    c = sp.add_parser("compare"); c.add_argument("--dir", required=True); c.add_argument("--layer", type=int, required=True)
    c.add_argument("--gemm", required=True); c.add_argument("--stock", required=True); c.add_argument("--stages"); c.add_argument("--rows")
    a = ap.parse_args()
    sys.exit(selftest() if a.cmd == "selftest" else compare(a))
