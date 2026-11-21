#!/usr/bin/env python3
"""Standalone verifier for native-r3-card2-agentic-20261121."""

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
    ife = json.loads((ROOT / "evidence/ifeval-full-paired.json").read_text(encoding="utf-8"))
    rep_i = ife["full_r3_lossless_official_report"]
    require(rep_i["arm_run_status"] == {"off": "finished", "on": "finished"}, "IFEval unfinished")
    for arm in ("off", "on"):
        require(rep_i["arm_case_counts"][arm]["request_error"] == 0, f"IFEval req error on {arm}")
        require(rep_i["arm_case_counts"][arm]["scoring_error"] == 0, f"IFEval score error on {arm}")
    bf = json.loads((ROOT / "evidence/bfcl-full-paired.json").read_text(encoding="utf-8"))
    rep_b = bf["full_r3_lossless_official_report"]
    require(rep_b["planned_pairs"] == 160 and rep_b["resolved_pairs"] == 160, "BFCL lossless unresolved pairs")
    wc = bf["full_wire_and_counts"]
    require(wc["lossless_serial"]["infra"] == 0 and wc["lossless_native"]["infra"] == 0, "BFCL lossless infra != 0")
    q = wc["q8c_native_partial"]
    require(q["completed_cases"] == 49 and q["infra"] == 111, "BFCL Q8C partial denominator mismatch")


def main():
    verify_sha256sums()
    verify_text_and_leaks()
    verify_packet()
    print("native-r3-card2-agentic-20261121 verification PASSED cleanly.")


if __name__ == "__main__":
    main()
