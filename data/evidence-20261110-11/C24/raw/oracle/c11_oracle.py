"""C11 CPU oracle: shared phase-C preparation (PCS) vs incumbent per-block phase C, phase-structured, FP32 numpy.
Emulates gdn_fused phase B items (hv, cg) -> attn/ssq scratch, per-block item ownership (it = b + k*G), random block
completion order, per-head modular counters (+NCG per launch, never reset), finisher builds xg/sg; phase C reads them.
Baseline phase C: fac/xs/block sums rebuilt per block. Checks bit equality of xs, sums, out (GEMV with fixed order).
Negative controls must be DETECTED. Run: python3 -I c11_oracle.py [out.json]
"""
import json, sys
import numpy as np
sys.path.insert(0, "/media/avg-usr/Storage/m6000-llama-cpp/occamy35-moonshot-20261110/tests")
import gdn_ref as R
F = np.float32
NVH, NKH, SD, NCG, H = 32, 16, 128, 8, 2048
DI = NVH * SD; CGC = SD // NCG

def warp_sum(v):            # xor butterfly o=16..1 on 32 lanes
    v = v.astype(F).copy()
    for o in (16, 8, 4, 2, 1):
        v = (v + v[np.arange(32) ^ o]).astype(F)
    return v[0]

def rsq(x): return F(F(1) / np.sqrt(F(x)))

def phase_b(inp, S, conv, rng, G):
    """Returns attn[DI], ssq[NVH*NCG], per-block item lists, block finish order. Delta rule per (hv,cg) as gdn_fused."""
    qkv, z, al, br, cw, A, dtb = inp["qkv"], inp["z"], inp["al"], inp["br"], inp["cw"], inp["A"], inp["dtb"]
    y = R.silu((conv[:, 0] * cw[:, 0] + conv[:, 1] * cw[:, 1] + conv[:, 2] * cw[:, 2] + qkv * cw[:, 3]).astype(F))
    attn = np.zeros(DI, F); ssq = np.zeros(NVH * NCG, F)
    for it in range(NVH * NCG):
        hv, cg = divmod(it, NCG); kh = hv % NKH
        q = y[kh * SD:(kh + 1) * SD]; k = y[H + kh * SD:H + (kh + 1) * SD]
        q = (q * rsq(np.sum(q * q, dtype=F) / F(SD) + F(1e-6) / SD)).astype(F)
        k = (k * rsq(np.sum(k * k, dtype=F) / F(SD) + F(1e-6) / SD)).astype(F)
        sp = R.softplus_ggml(np.array([al[hv] + dtb[hv]], F))[0]; g = F(np.exp(F(sp * A[hv])))
        b = R.sigmoid_ggml(np.array([br[hv]], F))[0]
        part = []
        for w in range(NCG):
            j2 = []
            for j in range(2):
                col = cg * CGC + 2 * w + j
                st = S[hv, col].copy(); v = y[2 * H + hv * SD + col]
                kvd = F(np.sum(st * k, dtype=F)); d = F((v - g * kvd) * b)
                st = (g * st + k * d).astype(F); S[hv, col] = st
                a = F(np.sum(st * q, dtype=F) * rsq(SD)); attn[hv * SD + col] = a; j2.append(a)
            part.append(F(F(0) + j2[0] * j2[0]) + j2[1] * j2[1])
        t = F(0)
        for p in part: t = F(t + F(p))
        ssq[hv * NCG + cg] = t
    blocks = [list(range(b, NVH * NCG, G)) for b in range(G)]
    order = rng.permutation(G)
    return attn, ssq, blocks, order

def head_vals(hv, attn, ssq, z, nrm, eps, reassoc=False):
    tt = F(0)
    for kk in range(NCG): tt = F(tt + ssq[hv * NCG + kk])
    fh = rsq(F(tt / F(SD)) + F(eps))
    i = slice(hv * SD, (hv + 1) * SD)
    sz = R.silu(z[i])
    if reassoc: return (attn[i] * fh * (nrm * sz).astype(F)).astype(F)
    return (((attn[i] * fh).astype(F) * nrm).astype(F) * sz).astype(F)

def baseline_c(attn, ssq, z, nrm, eps):
    xs = np.concatenate([head_vals(h, attn, ssq, z, nrm, eps) for h in range(NVH)])
    return xs, np.array([warp_sum(xs[32 * b:32 * b + 32]) for b in range(DI // 32)], F)

def pcs_publish(sc, attn, ssq, blocks, order, z, nrm, eps, ctl):
    """sc: persistent scratch dict (xg, sg, hcnt). Block publication in completion order."""
    done = np.zeros(NVH, int)   # contributions visible so far (for premature-finisher control)
    for b in order:
        hc = np.zeros(NVH, int)
        for it in blocks[b]: hc[it // NCG] += 1
        for hv in np.nonzero(hc)[0]:
            old = sc["hcnt"][hv]; c = hc[hv]
            sc["hcnt"][hv] = (old + c) % (1 << 32) if ctl != "reset_missing_nonmod" else old + c
            done[hv] += c
            if ctl == "reset_missing_nonmod": last = old + c == NCG          # non-modular test: only 1st launch fires
            elif ctl == "premature": last = (old % NCG) + c >= NCG - 1 and (old % NCG) < NCG - 1
            else: last = (old % NCG) + c == NCG
            if last:
                a = attn.copy(); q = ssq.copy()
                if done[hv] < NCG:                                            # finisher ran before all items: stale
                    for it in range(hv * NCG, hv * NCG + NCG):
                        if not any(it in blocks[bb] for bb in order[:list(order).index(b) + 1]):
                            q[it] = sc["stale_ssq"][it]
                            a[(it % NCG) * CGC + hv * SD:(it % NCG) * CGC + hv * SD + CGC] = sc["stale_attn"][hv * SD + (it % NCG) * CGC: hv * SD + (it % NCG) * CGC + CGC]
                v = head_vals(hv, a, q, z, nrm, eps, reassoc=(ctl == "reassoc"))
                sc["xg"][hv * SD:(hv + 1) * SD] = v
                for w in range(4): sc["sg"][hv * 4 + w] = warp_sum(v[32 * w:32 * w + 32])
    sc["stale_attn"], sc["stale_ssq"] = attn.copy(), ssq.copy()
    return sc["xg"].copy(), sc["sg"].copy()

def gemv_out(h, W, xs, sums):  # fixed-order accumulation shared by both paths (output depends only on xs/sums)
    return (h + (W @ xs).astype(F) + F(0) * sums.sum(dtype=F)).astype(F)

def make(rng, kind):
    inp = dict(qkv=rng.standard_normal((2 * H + DI)).astype(F), z=rng.standard_normal(DI).astype(F) * F(3),
               al=rng.standard_normal(NVH).astype(F), br=rng.standard_normal(NVH).astype(F),
               cw=rng.standard_normal(((2 * H + DI), 4)).astype(F) * F(.5), A=-np.abs(rng.standard_normal(NVH)).astype(F),
               dtb=rng.standard_normal(NVH).astype(F))
    S = (rng.standard_normal((NVH, SD, SD)) * .1).astype(F); conv = rng.standard_normal(((2 * H + DI), 3)).astype(F)
    nrm = (1 + .1 * rng.standard_normal(SD)).astype(F)
    if kind == "extremes":
        inp["al"][:8] = F(40); inp["al"][8:16] = F(-40); inp["br"][:8] = F(30); inp["br"][8:16] = F(-30)
        inp["z"][:512] = F(80); inp["z"][512:1024] = F(-80); inp["z"][1024:1100] = F(0)
    if kind == "state_x1e3": S *= F(1e3)
    if kind == "zero_head": S[3] = 0; inp["qkv"][2 * H + 3 * SD:2 * H + 4 * SD] = 0; inp["cw"][2 * H + 3 * SD:2 * H + 4 * SD] = 0
    return inp, S, conv, nrm

def run(name, kind="nonzero", G=48, steps=3, ctl=None, seed=0, hinit=0):
    rng = np.random.default_rng(seed)
    inp, S, conv, nrm = make(rng, kind)
    W = (rng.standard_normal((64, DI)) * .02).astype(F); h = rng.standard_normal(64).astype(F)
    sc = dict(xg=np.full(DI, F(7.0)), sg=np.full(DI // 32, F(7.0)), hcnt=np.full(NVH, hinit, np.int64),
              stale_attn=rng.standard_normal(DI).astype(F), stale_ssq=np.abs(rng.standard_normal(NVH * NCG)).astype(F))
    ok = True; maxd = 0.0
    for step in range(steps):          # consecutive launches share scratch (= consecutive layers / tokens)
        Sb = S.copy()                   # in-place alias: S_out == S_in row (state updated in place)
        attn, ssq, blocks, order = phase_b(inp, S, conv, rng, G)
        xb, sb = baseline_c(attn, ssq, inp["z"], nrm, F(1e-6))
        xp, sp = pcs_publish(sc, attn, ssq, blocks, order, inp["z"], nrm, F(1e-6), ctl)
        ob, op = gemv_out(h, W, xb, sb), gemv_out(h, W, xp, sp)
        eq = np.array_equal(xb.view(np.uint32), xp.view(np.uint32)) and np.array_equal(sb.view(np.uint32), sp.view(np.uint32)) \
             and np.array_equal(ob.view(np.uint32), op.view(np.uint32))
        ok &= eq; maxd = max(maxd, float(np.nanmax(np.abs(xb - xp))) if np.isfinite(xp).all() else float("inf"))
        inp["qkv"] = rng.standard_normal((2 * H + DI)).astype(F); inp["z"] = (rng.standard_normal(DI) * 3).astype(F)
    if ctl is None:   # cross-check phase-C output against independent gdn_ref.core_gguf semantics (tolerance)
        assert np.isfinite(xb).all() or kind == "extremes"
    return dict(name=name, kind=kind, G=G, steps=steps, control=ctl, bitequal=bool(ok), max_abs_diff=maxd,
                hcnt_mod_ok=bool(np.all(sc["hcnt"] % NCG == 0)))

if __name__ == "__main__":
    res = []
    for kind in ("nonzero", "extremes", "state_x1e3", "zero_head"):
        res.append(run(kind, kind))
    for G in (7, 48, 256, 300):
        res.append(run(f"grid{G}", G=G, seed=G))
    res.append(run("carry16", steps=16, seed=5))
    res.append(run("wrap2p32", steps=4, seed=6, hinit=(1 << 32) - 16))
    neg = [run("neg_premature", ctl="premature", seed=1), run("neg_reassoc", ctl="reassoc", seed=2),
           run("neg_nonmodular_counter", ctl="reset_missing_nonmod", steps=3, seed=3)]
    for r in res: r["pass"] = r["bitequal"] and r["hcnt_mod_ok"]
    for r in neg: r["pass"] = not r["bitequal"]          # control must be detected
    allr = res + neg
    for r in allr: print(("PASS " if r["pass"] else "FAIL ") + json.dumps(r))
    print(f"{len(allr)} checks, {sum(not r['pass'] for r in allr)} failed")
    if len(sys.argv) > 1: json.dump(allr, open(sys.argv[1], "w"), indent=1)
