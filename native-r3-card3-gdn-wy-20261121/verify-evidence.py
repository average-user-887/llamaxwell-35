#!/usr/bin/env python3
"""Standalone verifier for native-r3-card3-gdn-wy-20261121."""

from hashlib import file_digest
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
LEAK_RE = re.compile(
    "(?:"
    + "/" + "home/" + "|" + "/" + "media/"
    + "|G" + "PU-"
    + "|127" + r"\.0\.0\.1"
    + "|192" + r"\.168\."
    + "|172" + r"\.21\."
    + "|avg" + "-usr"
    + "|r3" + "bench-"
    + "|r3" + "card"
    + "|r3" + "wy-"
    + "|nm" + "moon-"
    + "|B550" + "-GAMING-X"
    + ")",
    re.IGNORECASE,
)


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


def verify_packet():
    res = json.loads((ROOT / "evidence/results.json").read_text(encoding="utf-8"))
    require(len(res["numerical_cases"]) == 11, "Expected 11 numerical cases")
    for r in res["numerical_cases"]:
        require(r["status"] == "PASS" and r["nonfinite_dst"] == 0 and r["nonfinite_st"] == 0, f"Case failed: {r}")
        require(r["ratio_dst"] <= 2.0 and r["wy_dst_rel_rmse"] <= 1e-4 and r["wy_st_rel_rmse"] <= 1e-4, f"Accuracy fail: {r}")
    require(len(res["timing_abba"]) == 2 and all(t["speedup_x"] < 1.0 for t in res["timing_abba"]), "Unexpected WY speedup")


def main():
    verify_sha256sums()
    verify_text_and_leaks()
    verify_packet()
    print("native-r3-card3-gdn-wy-20261121 verification PASSED cleanly.")


if __name__ == "__main__":
    main()
