#!/usr/bin/env python3
"""Standalone verifier for native-r3-card1-verified-20261121."""

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
    res = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    require(res["t123_all_passed"] is True, "t123 failed")
    w = res["warm80k_means"]
    require(w["exact"]["cold_exact_vs_r2"] == [True, True] and w["exact"]["warm_exact_vs_r2"] == [True, True], "exact parity mismatch")
    require(w["exact"]["warm_toks_s"] > w["r2"]["warm_toks_s"], "exact did not beat r2 warm")
    o = res["ordinary80_means"]
    require(o["exact"]["exact_stream_ids_vs_r2"] == [True, True], "ord80 exact parity mismatch")
    w_full = json.loads((ROOT / "evidence/warm80k-ordinary80-counterbalanced.json").read_text(encoding="utf-8"))
    cg = w_full["warm80k_runs"]["w80-exact-1"]["cuda_graphs"]["r3_cuda_graphs"]
    require(cg["captures"] > 0 and cg["replays"] > 0 and cg["incompat"] == 0, "80K CUDA graph check failed")


def main():
    verify_sha256sums()
    verify_text_and_leaks()
    verify_packet()
    print("native-r3-card1-verified-20261121 verification PASSED cleanly.")


if __name__ == "__main__":
    main()
