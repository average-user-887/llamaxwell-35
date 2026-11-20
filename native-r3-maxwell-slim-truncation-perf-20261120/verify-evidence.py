#!/usr/bin/env python3
"""Standalone verifier for native-r3-maxwell-slim-truncation-perf-20261120."""

from hashlib import file_digest
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
LEAK_RE = re.compile(r"/(?:home|media)/|" + "G" + "PU-" + r"|127\.0\.0\.1|" + "nm" + "moon-|" + "r3" + "bench-")


def require(cond, msg):
    if not cond:
        raise ValueError(msg)


def verify_sha256sums():
    sums = ROOT / "SHA256SUMS"
    require(sums.is_file(), "Missing SHA256SUMS")
    listed = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("  ", 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]), f"Bad line: {line}")
        rel = Path(fields[1])
        require(not rel.is_absolute() and ".." not in rel.parts and rel not in listed, f"Bad path: {fields[1]}")
        target = ROOT / rel
        require(target.is_file(), f"Missing file: {rel}")
        with target.open("rb") as f:
            require(file_digest(f, "sha256").hexdigest() == fields[0], f"SHA mismatch: {rel}")
        listed.add(rel)
    actual = {p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()} - {Path("SHA256SUMS")}
    require(listed == actual, f"File set mismatch: missing={actual - listed}, extra={listed - actual}")


def verify_text_and_leaks():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        require(text.isascii(), f"Non-ASCII bytes in {path.relative_to(ROOT)}")
        m = LEAK_RE.search(text)
        if m:
            raise ValueError(f"Sanitization leak {m.group(0)!r} in {path.relative_to(ROOT)}")


def verify_results():
    res = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    g6 = res["gate6_combined_r3_pp"]
    require(g6["last_query_type5_observed"] and g6["logit_compact_observed"], "gate6 missing type=5 or logit=compact")
    for name, chk in g6["checks"].items():
        require(chk["passed"] is True, f"gate6 check failed: {name}")
    act = g6["checks"]["CardA_and_A2_active"]
    require(act["compact"]["compact_only"] == 119 and act["draft_compact"]["compact_only"] == 119, "gate6 active compact count mismatch")
    require(act["draft_compact"]["fallback"] == 0 and act["exact_parity_vs_shadow"] is True, "gate6 fallback/parity mismatch")
    require(res["warm80k_sweep"]["r3_exact_lq"]["parity_vs_r2_native"]["warm_exact"] is True, "r3_exact_lq parity failed")
    require(res["ordinary80_sweep"]["runs"]["ord80_exact_ub1024_v4"]["exact_stream_ids_vs_r2"] is True, "ord80_exact_ub1024_v4 parity failed")


def main():
    verify_sha256sums()
    verify_text_and_leaks()
    verify_results()
    print("R3 packet verification PASSED cleanly.")


if __name__ == "__main__":
    main()
