"""C10 CPU oracle: staged q/k + gate reuse in phase B (GGML_CUDA_OCCAMY_GDN_QKREUSE) vs baseline item schedule.
usage: cd /tmp && python3 -I c10_oracle.py <WORK>/tests <outdir>
Kernel-faithful per-item arithmetic (same op order in both arms), so baseline-vs-reuse must be BIT EQUAL in numpy;
both arms are also checked vs gdn_ref.core_gguf (tolerance; different formula order).
Blocks of grid G run INTERLEAVED in random order (one item step per pick); conv/ssm rows alias in place (or differ);
q/k conv state is written in phase C. Negative controls: stale cache (never re-tag), early q/k conv write."""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import gdn_ref as R
F = np.float32
NK, VPK, HD, NCG = 16, 2, 128, 8
NV, QK, CGC = NK * VPK, NK * HD, HD // NCG
EPS_L2, SCALE_L2, EPS_G = F(1e-6 / HD), F(1 / np.sqrt(HD)), F(1e-6)   # ggml l2_norm as rms_norm(eps/n)*1/sqrt(n)

# ---- fixtures (copied from tests/gdn_selftest.py mk/hf_to_gguf_inputs; that file runs on import) ----
def mk(seed, state_scale=0.1):
    rng = np.random.default_rng(seed); C = 2 * QK + NV * HD
    f = lambda *s: rng.standard_normal(s).astype(F)
    d = dict(qkv=f(C), z=f(NV * HD), alpha=f(NV), beta=f(NV), conv_state=f(C, 3) * F(0.5), conv_w=f(C, 4) * F(0.5),
             A=-np.exp(f(NV) * F(0.5)), dt_bias=f(NV) * F(0.5), ssm_norm=(F(1) + f(HD) * F(0.1)).astype(F), S=f(NV, HD, HD) * F(state_scale))
    for h in range(NV):
        d["qkv"][2 * QK + h * HD:2 * QK + (h + 1) * HD] += F(3 * (h + 1)); d["z"][h * HD:(h + 1) * HD] += F(0.1 * h); d["S"][h] += F(0.01 * (h + 1))
    d["alpha"][:] += np.arange(NV, dtype=F) * F(0.1); d["beta"][:] += np.arange(NV, dtype=F) * F(0.05)
    for kh in range(NK): d["qkv"][kh * HD:(kh + 1) * HD] += F(kh + 1); d["qkv"][QK + kh * HD:QK + (kh + 1) * HD] -= F(0.5 * (kh + 1))
    return d

def to_gguf(d):
    p = R.hf_to_gguf_perm(NK, VPK); g = dict(d)
    for k in ("qkv", "conv_state", "conv_w"): g[k] = R.to_gguf_channels(d[k], NK, VPK, HD)
    g["z"] = d["z"].reshape(NV, HD)[p].reshape(-1)
    for k in ("alpha", "beta", "A", "dt_bias"): g[k] = d[k][p]
    g["M"] = d["S"][p].transpose(0, 2, 1).copy()
    return g

def ext_gate(d):
    d["alpha"][:] = np.resize(np.array([-100, -30, -1, 0, 19.99, 20.01, 25, 100], F), NV); d["dt_bias"][:] = 0
    d["beta"][:] = np.resize(np.array([-100, -20, -1, 0, 1, 20, 100, 1e4], F), NV)
def ext_qk(d):
    d["qkv"][0:128] = 0; d["qkv"][128:256] *= F(1e-20); d["qkv"][256:384] *= F(1e18)
    d["qkv"][QK:QK + 128] = 0; d["qkv"][QK + 128:QK + 256] *= F(1e-20); d["qkv"][QK + 256:QK + 384] *= F(1e18); d["conv_w"][:QK * 2] *= F(10)
def big_state(d): d["S"] *= F(1e3)

# ---- kernel-faithful emulation of gdn_fused phase B/C with item schedule + block cache ----
def items_for_block(b, G, rqk):
    n = NV * NCG
    if not rqk: return [(it // NCG, it % NCG) for it in range(b, n, G)]
    lo, hi = (b * n) // G, ((b + 1) * n) // G
    return [(it // (2 * NCG) + NK * ((it // NCG) & 1), it % NCG) for it in range(lo, hi)]

def emul(g, G, rqk, seed, conv_buf, conv_row, conv_out_row, ssm_buf, ssm_row, ssm_out_row, stale_bug=False, early_qk=False, count=None):
    qkv, w = g["qkv"], g["conv_w"]
    rng = np.random.default_rng(seed)
    queues = [items_for_block(b, G, rqk) for b in range(G)]
    cache = [dict(kh=-1, hv=-1) for _ in range(G)]
    attn = np.zeros((NV, HD), F); ssq = np.zeros((NV, NCG), F)
    def conv(cs):
        ci = conv_buf[conv_row][cs].copy()                              # read old state (row may alias out row)
        y = (ci[:, 0] * w[cs, 0] + ci[:, 1] * w[cs, 1] + ci[:, 2] * w[cs, 2] + qkv[cs] * w[cs, 3]).astype(F)
        return R.silu(y), ci
    while any(queues):
        b = rng.choice([i for i, q in enumerate(queues) if q]); hv, cg = queues[b].pop(0); kh = hv % NK; cb = cache[b]
        need_qk = (not rqk) or (kh != cb["kh"] and not (stale_bug and cb["kh"] >= 0))
        if need_qk:
            cs = np.r_[kh * HD:(kh + 1) * HD, QK + kh * HD:QK + (kh + 1) * HD]
            y, _ = conv(cs)
            red = (y * y).reshape(8, 32).sum(1, dtype=F)                # per-warp sums (order != GPU shuffle; same in both arms)
            cb["qn"], cb["kn"] = y[:HD].copy(), y[HD:].copy()
            cb["qs"] = F(F(1) / np.sqrt(F(red[0] + red[1] + red[2] + red[3]) / F(HD) + EPS_L2)) * SCALE_L2
            cb["ks"] = F(F(1) / np.sqrt(F(red[4] + red[5] + red[6] + red[7]) / F(HD) + EPS_L2)) * SCALE_L2
            cb["kh"] = kh
            if count is not None: count["qk"] += 1
            if early_qk:
                for c in cs: conv_buf[conv_out_row][c] = (conv_buf[conv_row][c][1], conv_buf[conv_row][c][2], qkv[c])
        cv = np.arange(2 * QK + hv * HD + cg * CGC, 2 * QK + hv * HD + (cg + 1) * CGC)
        vv, ci = conv(cv)                                               # unique owner: read then write
        conv_buf[conv_out_row][cv] = np.stack([ci[:, 1], ci[:, 2], qkv[cv]], 1)
        if (not rqk) or hv != cb["hv"]:
            al = F(g["alpha"][hv] + g["dt_bias"][hv])
            sp = R.softplus_ggml(np.array([al], F))[0]
            with np.errstate(over="ignore"):
                cb["gexp"] = F(np.exp(F(sp * g["A"][hv]))); cb["beta"] = R.sigmoid_ggml(np.array([g["beta"][hv]], F))[0]
            cb["hv"] = hv
            if count is not None: count["gate"] += 1
        qv, kv, gexp, beta = cb["qn"] * cb["qs"], cb["kn"] * cb["ks"], cb["gexp"], cb["beta"]
        for j in range(CGC):
            col = cg * CGC + j
            a = ssm_buf[ssm_row, hv, col].copy()
            kvd = F(np.sum(a * kv, dtype=F)); delta = F((vv[j] - gexp * kvd) * beta)
            a = (gexp * a + kv * delta).astype(F)
            att = F(F(np.sum(a * qv, dtype=F)) * F(1 / np.sqrt(F(HD))))
            ssm_buf[ssm_out_row, hv, col] = a; attn[hv, col] = att; ssq[hv, cg] += F(att * att)
    if not early_qk:                                                    # phase C q/k conv state write
        cin = conv_buf[conv_row][:2 * QK].copy()
        conv_buf[conv_out_row][:2 * QK] = np.stack([cin[:, 1], cin[:, 2], qkv[:2 * QK]], 1)
    with np.errstate(over="ignore", divide="ignore"):
        fac = (F(1) / np.sqrt(ssq.sum(1, dtype=F) / F(HD) + EPS_G)).astype(F)
        return (attn * fac[:, None] * g["ssm_norm"][None] * R.silu(g["z"].reshape(NV, HD))).astype(F).ravel()

def bufs(g, in_row, rows=3):
    C = g["qkv"].size
    cb = np.random.default_rng(99).standard_normal((rows, C, 3)).astype(F); sb = np.random.default_rng(98).standard_normal((rows, NV, HD, HD)).astype(F)
    cb[in_row] = g["conv_state"]; sb[in_row] = g["M"]; return cb, sb

def bits(a, b): a = np.asarray(a, F).ravel(); b = np.asarray(b, F).ravel(); return int((a.view(np.uint32) != b.view(np.uint32)).sum())
def scaled(a, b):
    a = np.asarray(a, F).ravel(); b = np.asarray(b, F).ravel()
    with np.errstate(invalid="ignore"): return float(np.nanmax(np.abs(a - b) / (np.abs(b) + F(1e-3) * np.nanmax(np.abs(b)))))

res, fails = {}, []
def run_arm(g, G, rqk, seed, in_row, out_row, **kw):
    cb, sb = bufs(g, in_row); keep = (cb[0].copy(), sb[0].copy()) if 0 not in (in_row, out_row) else None
    cnt = dict(qk=0, gate=0)
    o = emul(g, G, rqk, seed, cb, in_row, out_row, sb, in_row, out_row, count=cnt, **kw)
    return o, cb[out_row], sb[out_row], cnt, (keep is None or (np.array_equal(cb[0], keep[0]) and np.array_equal(sb[0], keep[1])))

cases = [("nonzero_state", 8, None), ("gate_extremes", 4, ext_gate), ("qk_zero_tiny_huge", 5, ext_qk), ("large_state", 6, big_state)]
for name, seed, mod in cases:
    d = mk(seed)
    if mod: mod(d)
    g = to_gguf(d); o_ref, M_ref, c_ref = R.core_gguf(g["qkv"], g["z"], g["alpha"], g["beta"], g["conv_state"], g["conv_w"], g["A"], g["dt_bias"], g["ssm_norm"], g["M"], NK, VPK, HD)
    for G, rows in [(48, (1, 1)), (48, (2, 0)), (7, (1, 1)), (256, (1, 1)), (300, (1, 1))]:
        if name != "nonzero_state" and (G, rows) not in [(48, (1, 1))]: continue
        key = f"{name}/G{G}/in{rows[0]}out{rows[1]}"
        oA, cA, sA, nA, uA = run_arm(g, G, False, seed, *rows)
        oB, cB, sB, nB, uB = run_arm(g, G, True, seed + 1000, *rows)
        r = dict(out_bits=bits(oB, oA), state_bits=bits(sB, sA), conv_bits=bits(cB, cA), untouched_row_ok=bool(uA and uB),
                 ref_out_scaled=scaled(oB, o_ref), ref_state_scaled=scaled(sB, M_ref), ref_conv_bits=bits(cB, c_ref),
                 nan_match=bool(np.array_equal(np.isnan(oA), np.isnan(oB))), qk_evals=(nA["qk"], nB["qk"]), gate_evals=(nA["gate"], nB["gate"]))
        r["ok"] = bool(r["out_bits"] == 0 and r["state_bits"] == 0 and r["conv_bits"] == 0 and r["untouched_row_ok"] and r["ref_conv_bits"] == 0
                       and r["ref_out_scaled"] <= 1e-3 and r["ref_state_scaled"] <= 1e-3)
        res[key] = r
        if not r["ok"]: fails.append(key)

# 16-step recurrence carry (state/conv of step t feed step t+1), in place, G=48, both arms
d = mk(7); g = to_gguf(d); rng = np.random.default_rng(70)
cbA, sbA = bufs(g, 1); cbB, sbB = cbA.copy(), sbA.copy(); worst = 0
for t in range(16):
    g.update(qkv=rng.standard_normal(g["qkv"].shape).astype(F), z=rng.standard_normal(g["z"].shape).astype(F),
             alpha=rng.standard_normal(NV).astype(F), beta=rng.standard_normal(NV).astype(F))
    oA = emul(g, 48, False, t, cbA, 1, 1, sbA, 1, 1); oB = emul(g, 48, True, t + 50, cbB, 1, 1, sbB, 1, 1)
    worst = max(worst, bits(oA, oB) + bits(sbA[1], sbB[1]) + bits(cbA[1], cbB[1]))
res["recurrence16/G48/total_bit_mismatch"] = dict(value=worst, ok=worst == 0)
if worst: fails.append("recurrence16")

# negative controls (must be DETECTED)
d = mk(8); g = to_gguf(d); o_ref, M_ref, c_ref = R.core_gguf(g["qkv"], g["z"], g["alpha"], g["beta"], g["conv_state"], g["conv_w"], g["A"], g["dt_bias"], g["ssm_norm"], g["M"], NK, VPK, HD)
# stale cache needs a grid whose chunks straddle a k head (G=48 chunks never do: 256/48 boundaries hit every 16th item)
for nm, G, kw in [("stale_cache_never_retag_G7", 7, dict(stale_bug=True)), ("early_qk_conv_write_with_reuse_G48", 48, dict(early_qk=True))]:
    oA, _, sA, _, _ = run_arm(g, G, False, 8, 1, 1)
    oX, _, sX, _, _ = run_arm(g, G, True, 9, 1, 1, **kw)
    r = dict(out_bits=bits(oX, oA), state_bits=bits(sX, sA), ref_out_scaled=scaled(oX, o_ref)); r["ok"] = bool(r["out_bits"] > 0 and r["ref_out_scaled"] > 1e-3)
    res["negative_control/" + nm] = r
    if not r["ok"]: fails.append(nm)

# work / byte model (per GDN layer, per token) for the real grid
def model(G):
    qk = sum(len({hv % NK for hv, _ in items_for_block(b, G, True)}) for b in range(G))
    gate = sum(len({hv for hv, _ in items_for_block(b, G, True)}) for b in range(G))
    per = [len(items_for_block(b, G, True)) for b in range(G)]; per0 = [len(items_for_block(b, G, False)) for b in range(G)]
    B = 32  # bytes per q/k channel evaluation: conv state 12 + conv_w 16 + qkv 4
    return dict(G=G, qk_head_evals_base=NV * NCG, qk_head_evals_reuse=qk, qk_bytes_base=NV * NCG * 256 * B, qk_bytes_reuse=qk * 256 * B,
                qk_unique_bytes=NK * 256 * B, gate_evals_base=NV * NCG, gate_evals_reuse=gate, items_per_block_base=(min(per0), max(per0)),
                items_per_block_reuse=(min(per), max(per)), state_bytes_rw=2 * NV * HD * HD * 4)
res["work_model"] = [model(G) for G in (48, 24, 72)]
res["_summary"] = dict(failed=fails, n_checks=sum(1 for v in res.values() if isinstance(v, dict) and "ok" in v))
json.dump(res, open(sys.argv[2] + "/c10_oracle.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else (int(x) if isinstance(x, np.integer) else str(x)))
print(json.dumps(res["_summary"])); print(json.dumps(res["work_model"][0]))
