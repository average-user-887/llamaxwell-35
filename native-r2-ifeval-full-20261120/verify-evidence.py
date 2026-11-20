#!/usr/bin/env python3
"""Standalone zero-dependency verifier for the public IFEval paired evidence packet."""

from hashlib import file_digest
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
STAGE_SPECS = {"smoke": (6, 12), "full": (541, 834)}
OUTCOMES = ("both_pass", "both_fail", "on_only", "off_only", "missing")
LEAK_PATTERNS = re.compile(r"/(?:home|media)/|" + "G" + "PU-" + r"|127\.0\.0\.1|" + "nm" + "moon-")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def verify_sha256sums():
    sums_path = ROOT / "SHA256SUMS"
    require(sums_path.is_file(), "Missing SHA256SUMS")
    listed = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("  ", 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                f"Malformed SHA256SUMS line: {line}")
        rel = Path(fields[1])
        require(not rel.is_absolute() and ".." not in rel.parts and rel not in listed,
                f"Unsafe or duplicate SHA256SUMS path: {fields[1]}")
        target = ROOT / rel
        require(target.is_file(), f"Missing listed file: {rel}")
        with target.open("rb") as handle:
            actual_hash = file_digest(handle, "sha256").hexdigest()
        require(actual_hash == fields[0], f"SHA-256 mismatch for {rel}")
        listed.add(rel)
    actual_files = {p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()}
    require(listed == actual_files - {Path("SHA256SUMS")},
            "SHA256SUMS does not cover exact package file set")


def verify_leaks():
    readme = ROOT / "README.md"
    require(readme.read_text(encoding="utf-8").isascii(), "README.md must be ASCII-only")
    for path in [readme, *sorted((ROOT / "evidence").glob("*.json"))]:
        text = path.read_text(encoding="utf-8")
        match = LEAK_PATTERNS.search(text)
        require(match is None, f"Forbidden private string in {path.name}: {match.group(0) if match else ''}")


def empty_tally():
    return {key: 0 for key in (*OUTCOMES, "off_pass", "off_missing", "on_pass", "on_missing")}


def add_counts(tally, off, on):
    for key, val in (("off", off), ("on", on)):
        tally[f"{key}_pass"] += val is True
        tally[f"{key}_missing"] += val is None
    if off is None or on is None:
        paired = "missing"
    elif off and on:
        paired = "both_pass"
    elif not off and not on:
        paired = "both_fail"
    else:
        paired = "off_only" if off else "on_only"
    tally[paired] += 1


def with_rates(tally):
    denom = sum(tally[k] for k in OUTCOMES)
    off_res = denom - tally["off_missing"]
    on_res = denom - tally["on_missing"]
    return {
        **tally,
        "planned": denom,
        "off_resolved": off_res,
        "on_resolved": on_res,
        "off_accuracy": tally["off_pass"] / denom if off_res == denom else None,
        "on_accuracy": tally["on_pass"] / denom if on_res == denom else None,
        "on_minus_off": ((tally["on_only"] - tally["off_only"]) / denom if not tally["missing"] else None),
    }


def verify_physical(physical, mode):
    require(physical["mode"] == mode, f"Physical mode mismatch: {physical['mode']}")
    compact = physical["compact"]
    guard = physical["compact_guard"]
    counters = physical["physical_counters"]
    require(min(compact.values()) >= 0 and min(guard.values()) >= 0 and min(counters.values()) >= 0,
            f"Negative counter in {mode}")
    require(compact["requested"] >= compact["compact_only"], f"Invalid compact_only in {mode}")
    offers = sum(guard.values()) - compact["compact_only"]
    require(offers >= 0 and physical["observed_fallback_proposals"] == offers,
            f"Fallback proposals mismatch in {mode}")
    require(physical["down4_launches"] == counters["down4"], f"down4_launches mismatch in {mode}")
    require(physical["down4_matches_40_times_guard_offers"] == (counters["down4"] == 40 * sum(guard.values())),
            f"down4_matches_40_times_guard_offers mismatch in {mode}")


def verify_scores(score):
    stage = score["stage"]
    require(stage in STAGE_SPECS, f"Unsupported stage: {stage}")
    expected_prompts, expected_insts = STAGE_SPECS[stage]
    require(all(v is True for v in score["same_build_checks"].values()), "same_build_checks failed")

    public_ids = score["public_ids"]
    require(len(public_ids) == len(set(public_ids)) == expected_prompts, "Prompt ID count/uniqueness failed")
    case_insts = score["case_instructions"]
    require(set(case_insts.keys()) == {str(i) for i in public_ids}, "case_instructions IDs mismatch")
    require(sum(len(v) for v in case_insts.values()) == expected_insts, "Instruction count mismatch")

    serial_cases = score["arms"]["serial"]["cases"]
    native_cases = score["arms"]["native"]["cases"]
    require(set(serial_cases.keys()) == set(native_cases.keys()) == {str(i) for i in public_ids},
            "Arm case keys mismatch public_ids")

    for mode in ("serial", "native"):
        verify_physical(score["arms"][mode]["physical"], mode)

    for mode_name in ("strict", "loose"):
        p_tally = empty_tally()
        i_tally = empty_tally()
        by_inst = {}
        for key in public_ids:
            skey = str(key)
            off = serial_cases[skey]
            on = native_cases[skey]
            inst_list = case_insts[skey]
            require(off["status"] in {"scored", "model_cap"} and on["status"] in {"scored", "model_cap"},
                    f"Unresolved IFEval case {skey}")
            if mode_name == "strict":
                for rec in (off, on):
                    require(len(rec["strict"]) == len(rec["loose"]) == len(inst_list),
                            f"Instruction length mismatch on {skey}")
                    require(not any(a and not b for a, b in zip(rec["strict"], rec["loose"], strict=True)),
                            f"strict pass failed loose on {skey}")
            av_list = off[mode_name]
            bv_list = on[mode_name]
            add_counts(p_tally, all(av_list), all(bv_list))
            for name, av, bv in zip(inst_list, av_list, bv_list, strict=True):
                add_counts(i_tally, av, bv)
                add_counts(by_inst.setdefault(name, empty_tally()), av, bv)

        require(with_rates(p_tally) == score["metrics"][mode_name]["prompt"],
                f"Prompt tally mismatch for {mode_name}")
        require(with_rates(i_tally) == score["metrics"][mode_name]["instruction"],
                f"Instruction tally mismatch for {mode_name}")
        expected_by_inst = {k: with_rates(v) for k, v in by_inst.items()}
        require(expected_by_inst == score["metrics"][mode_name]["by_instruction"],
                f"by_instruction mismatch for {mode_name}")


def verify_timing(score, timing):
    for mode in ("serial", "native"):
        arm_t = timing["arms"][mode]
        cases = arm_t["cases"]
        s_cases = score["arms"][mode]["cases"]
        require([c["id"] for c in cases] == score["public_ids"], f"{mode} timing IDs mismatch")
        for c in cases:
            sc = s_cases[str(c["id"])]
            require(c["status"] == sc["status"], f"{mode} timing status mismatch on {c['id']}")
        require(math.isclose(sum(c["wall_s"] for c in cases), arm_t["case_wall_s_sum"], rel_tol=0, abs_tol=1e-9),
                f"{mode} case_wall_s_sum mismatch")
        require(math.isclose(max(c["wall_s"] for c in cases), arm_t["max_case_wall_s"], rel_tol=0, abs_tol=1e-9),
                f"{mode} max_case_wall_s mismatch")
        require(sum(c["prompt_tokens"] for c in cases) == arm_t["prompt_tokens_sum"],
                f"{mode} prompt_tokens_sum mismatch")
        require(sum(c["completion_tokens"] for c in cases) == arm_t["completion_tokens_sum"],
                f"{mode} completion_tokens_sum mismatch")


def verify_launch(score, launch):
    for mode in ("serial", "native"):
        entry = launch[mode]
        arm = score["arms"][mode]
        obs_status = arm.get("observation_status", "nominal")
        require(obs_status in ("nominal", "revalidated_after_log_rotation"),
                f"Invalid observation_status for {mode}")
        require(entry.get("observation_status") == obs_status, f"Launch observation_status mismatch for {mode}")
        require(entry["completed_cases"] == len(score["public_ids"]), f"Launch completed_cases mismatch for {mode}")
        require(entry["cleanup_exit"] == arm["cleanup_exit"] == 0, f"Launch cleanup_exit mismatch for {mode}")
        require(entry["evaluator_exit"] == arm["evaluator_exit"] == 0, f"Launch evaluator_exit mismatch for {mode}")
        if obs_status == "revalidated_after_log_rotation":
            require(entry["runner_exit"] == arm["runner_exit"] in (0, 1),
                    f"Launch runner_exit mismatch for revalidated {mode}")
            require("observation_revalidation" in arm, f"Missing observation_revalidation for {mode}")
        else:
            require(entry["runner_exit"] == arm["runner_exit"] == 0, f"Launch runner_exit mismatch for {mode}")


def main():
    verify_sha256sums()
    verify_leaks()
    score = read_json("evidence/ifeval-full-score.json")
    timing = read_json("evidence/ifeval-full-timing.json")
    launch = read_json("evidence/launch-outcomes.json")
    verify_scores(score)
    verify_timing(score, timing)
    verify_launch(score, launch)
    print(f"Verified {score['stage']} IFEval paired evidence "
          f"({score['instruction_counts']['planned_prompts']} prompts, "
          f"{score['instruction_counts']['planned_instructions']} instructions), "
          f"physical counters, timing, launch outcomes, and SHA256SUMS.")


if __name__ == "__main__":
    main()
