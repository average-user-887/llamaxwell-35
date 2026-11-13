#!/usr/bin/env python3
"""Run ref_fa64.py compare across all N4 stops/layers and write N4/RESULT.md."""
import json
import os
import subprocess

M = "<LAB>"
NC3 = M + "/occamy35-night-20261113/NC3"
N4 = M + "/occamy35-night-20261113/N4"

STOPS = [4096, 4160, 32768, 33024, 98048, 98304]
LAYERS = [3, 19, 39]

rows_md = []
all_results = []

for s in STOPS:
    for l in LAYERS:
        ddir = f"{NC3}/dump/stop{s}"
        gfile = f"{NC3}/out_gemm/stop{s}/L{l}_O.f32"
        sfile = f"{NC3}/out_stock/stop{s}/L{l}_O.f32"
        stdir = f"{NC3}/out_gemm/stop{s}"
        if not (os.path.exists(gfile) and os.path.exists(sfile)):
            continue
        p = subprocess.run(
            [
                "python3", "-I", f"{NC3}/ref_fa64.py", "compare",
                "--dir", ddir, "--layer", str(l),
                "--gemm", gfile, "--stock", sfile, "--stages", stdir,
            ],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            rows_md.append(f"| {s} | {l} | ERR rc={p.returncode} | - | - | - | - |")
            continue
        j = json.loads(p.stdout)
        j["stop"] = s
        j["layer"] = l
        all_results.append(j)
        d = j.get("distance_vs_ref", {})
        gr = d.get("gemm_rmse", float("nan"))
        sr = d.get("stock_rmse", float("nan"))
        ratio = d.get("ratio_gemm_over_stock", float("nan"))
        closer = d.get("gemm_closer_frac", float("nan"))
        insitu = j.get("insitu_vs_stock_driver", {}).get("bit_equal_frac", float("nan"))
        rows_md.append(
            f"| {s} | {l} | {gr:.3e} | {sr:.3e} | {ratio:.3f} | {closer:.3f} | {insitu:.4f} |"
        )

json.dump(all_results, open(f"{N4}/compare.json", "w"), indent=2)

md = [
    "# N4: GEMM-prefill vs stock FA numerics bisect (against FP64 reference)",
    "",
    "| stop | layer | GEMM RMSE | stock RMSE | ratio (G/S) | GEMM closer frac | in-situ == stock drv |",
    "|---:|---:|---:|---:|---:|---:|---:|",
] + rows_md + [""]

open(f"{N4}/RESULT.md", "w").write("\n".join(md))
print(f"Wrote {N4}/RESULT.md ({len(all_results)} cells)")
