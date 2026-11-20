#!/usr/bin/env python3
"""Standalone verifier for the sanitized Token 170 evidence packet."""

from hashlib import file_digest
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
LEAK_PATTERNS = re.compile(r"/(?:home|media)/|" + "G" + "PU-" + r"|127\.0\.0\.1|" + "nm" + "moon-")


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def verify_sha256sums():
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed = set()
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2 and HEX64_RE.match(parts[0]), f"Bad SHA256SUMS line: {line}")
        rel = Path(parts[1])
        require(not rel.is_absolute() and ".." not in rel.parts and rel not in listed, f"Bad path: {parts[1]}")
        target = ROOT / rel
        require(target.is_file(), f"Missing file: {rel}")
        with target.open("rb") as h:
            actual = file_digest(h, "sha256").hexdigest()
        require(actual == parts[0], f"Hash mismatch for {rel}")
        listed.add(rel)
    actual_files = {p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()}
    require(listed == actual_files - {Path("SHA256SUMS")}, "SHA256SUMS file set mismatch")


def verify_leaks():
    readme = ROOT / "README.md"
    require(readme.read_text(encoding="utf-8").isascii(), "README.md must be ASCII-only")
    for path in [readme, *sorted((ROOT / "evidence").glob("*.json"))]:
        text = path.read_text(encoding="utf-8")
        require(text.isascii(), f"Non-ASCII content in {path.name}")
        m = LEAK_PATTERNS.search(text)
        require(m is None, f"Forbidden pattern in {path.name}: {m.group(0) if m else ''}")


def verify_packet():
    summary = json.loads((ROOT / "evidence/token170-summary.json").read_text(encoding="utf-8"))
    decisions_doc = json.loads((ROOT / "evidence/token170-decisions.json").read_text(encoding="utf-8"))
    launches = json.loads((ROOT / "evidence/launch-outcomes.json").read_text(encoding="utf-8"))

    require(summary["observation_seals_verified"] is True, "Observation seals must be verified")
    require(all(v is True for v in summary["same_input_and_build_checks"].values()), "Same input/build check failed")

    for mode in ("serial", "native"):
        arm = summary["arms"][mode]
        require(arm["cleanup_exit"] == 0, f"{mode} cleanup_exit nonzero")
        if arm.get("observation_status") == "revalidated_terminal_destructor_histogram":
            require(mode == "native" and arm["runner_exit"] == 1, f"{mode} unexpected revalidation exit")
            require(arm["physical"]["final_mv1_host"] == {"rounds": 101, "restores": 0, "acc_0": 12, "acc_1": 89},
                    "Native terminal mv1_host histogram mismatch")
        else:
            require(arm["observation_status"] == "nominal" and arm["runner_exit"] == 0,
                    f"{mode} runner_exit nonzero")
        phys = arm["physical"]
        offered = phys["offered_draft_tokens"]
        for key, coeff in phys["coefficients_expected"].items():
            require(phys["counters"][key] == coeff * offered, f"{mode} counter {key} ratio mismatch")
        if mode == "native":
            snap = phys["sampler_snapshot"]
            require(snap is not None and snap["elided"] == offered, "Native sampler_snapshot elided mismatch")
            require(snap["elided_rejected"] == offered - phys["accepted_draft_tokens"], "Native elided_rejected mismatch")
            require(snap["kept"] == 0 and snap["restored"] == 0, "Native kept/restored must be 0")

    records = decisions_doc["decisions"]
    require(len(records) == 170, f"Expected 170 decisions, got {len(records)}")
    require([r["ordinal"] for r in records] == list(range(1, 171)), "Ordinals must be 1..170")
    for r in records:
        require(len(r["serial"]["top8_logprobs"]) == 8 and len(r["native"]["top8_logprobs"]) == 8,
                f"Expected 8 logprobs at ordinal {r['ordinal']}")
        require(not any(k in r or k in r["serial"] or k in r["native"] for k in ("token_id", "token_ids", "text", "prompt")),
                f"Forbidden raw token field at ordinal {r['ordinal']}")

    rej_ord = summary["arms"]["native"]["result"]["first_rejection"]["decision_ordinal"]
    require(records == decisions_doc["decisions"], "Sanity")
    first_score_170 = next((r["ordinal"] for r in records if not r["top8_scores_equal"]), None)
    first_order_170 = next((r["ordinal"] for r in records if not r["top8_order_equal"]), None)
    first_emit_170 = next((r["ordinal"] for r in records if not r["emitted_equal"]), None)
    t170 = summary["compare"]["through_170"]
    require(t170["first_rounded_score_or_rank_difference"] == first_score_170, "through_170 score diff mismatch")
    require(t170["first_order_difference"] == first_order_170, "through_170 order diff mismatch")
    require(t170["first_emitted_difference"] == first_emit_170, "through_170 emitted diff mismatch")

    pre_records = [r for r in records if r["ordinal"] < rej_ord]
    pre = summary["compare"]["strictly_before_first_rejected_round"]
    require(pre["ordinals"] == [1, rej_ord - 1], "pre-rejection range mismatch")
    require(pre["first_rounded_score_or_rank_difference"] == next((r["ordinal"] for r in pre_records if not r["top8_scores_equal"]), None),
            "pre-rejection score diff mismatch")
    require(pre["first_order_difference"] == next((r["ordinal"] for r in pre_records if not r["top8_order_equal"]), None),
            "pre-rejection order diff mismatch")
    require(pre["first_emitted_difference"] == next((r["ordinal"] for r in pre_records if not r["emitted_equal"]), None),
            "pre-rejection emitted diff mismatch")

    agg = summary["decision_window_aggregates"]
    require(agg["through_170"]["scores_diff_count"] == sum(1 for r in records if not r["top8_scores_equal"]),
            "scores_diff_count mismatch")
    require(agg["through_170"]["order_diff_count"] == sum(1 for r in records if not r["top8_order_equal"]),
            "order_diff_count mismatch")
    require(agg["through_170"]["top1_flip_count"] == sum(1 for r in records if not r["top1_token_equal"]),
            "top1_flip_count mismatch")

    tc = summary["token_comparison"]
    if first_emit_170 is not None:
        require(tc["first_emitted_difference_1_based"] == first_emit_170, "Token comparison mismatch")
        require(tc["token_equality_prefix_len"] == first_emit_170 - 1, "Prefix len mismatch")

    require(len(launches["runs"]) >= 2, "Expected at least 2 arm entries in launch-outcomes.json")


def main():
    verify_sha256sums()
    verify_leaks()
    verify_packet()
    print("Verified Token 170 evidence packet, decision 1..170 recomputation, physical counters, and SHA256SUMS.")


if __name__ == "__main__":
    main()
