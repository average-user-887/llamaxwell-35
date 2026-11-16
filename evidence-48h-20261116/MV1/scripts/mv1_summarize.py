#!/usr/bin/env python3
# Summarise an MV1 run dir into measurements/ (no token ids, no prompts). usage: mv1_summarize.py RUN_DIR OUT_DIR
import json, glob, os, sys, shutil
run, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
def toks(d, s): return json.load(open(f"{run}/{d}/result.json"))["slots"][s]["out_tokens"]
def fdiv(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]: return i
    return "" if len(a) == len(b) else min(len(a), len(b))
ref = toks("np1_A_12761", "0")
rows = ["arm,kind,np,slot,prompt_tps,gen_tps,draft_n,draft_acc,t_first_s,latency_s,peak_vram_MiB,first_token_diff_vs_np1_A"]
for d in sorted(os.listdir(run)):
    p = f"{run}/{d}/result.json"
    if not os.path.exists(p): continue
    r = json.load(open(p)); kind = "B" if "_B" in d else "A"
    vr = [int(x) for x in open(f"{run}/{d}/vram.csv").read().split() if x.isdigit()]
    for s, v in sorted(r["slots"].items()):
        t = v.get("timings", {})
        rows.append(",".join(str(x) for x in [d, kind, r["np"], s, round(t.get("prompt_per_second", 0), 1), round(t.get("predicted_per_second", 0), 2), t.get("draft_n", ""), t.get("draft_n_accepted", ""), v["t_first_s"], v["latency_s"], max(vr) if vr else "", fdiv(ref, v["out_tokens"]) if r["np"] == 1 and r["sampling"].startswith("greedy") else ""]))
    os.makedirs(f"{out}/{d}", exist_ok=True)
    for f in ("mv1_counters.txt", "vram.csv"):
        if os.path.exists(f"{run}/{d}/{f}"): shutil.copy(f"{run}/{d}/{f}", f"{out}/{d}/{f}")
    json.dump({k: v for k, v in r.items() if k not in ("slots", "props_default_gen")} | {"slots": {s: {k: x for k, x in v.items() if k not in ("out_tokens", "ev")} for s, v in r["slots"].items()}}, open(f"{out}/{d}/result_summary.json", "w"), indent=1)
open(f"{out}/arms.csv", "w").write("\n".join(rows) + "\n")
shutil.copy(f"{run}/steps.log", f"{out}/steps.log")
if os.path.exists(f"{run}/nt/trace_B.nt.tsv"): shutil.copy(f"{run}/nt/trace_B.nt.tsv", f"{out}/trace_B.nt.tsv")
