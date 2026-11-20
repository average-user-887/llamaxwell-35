#!/usr/bin/env python3
"""Standalone zero-dependency verifier for the public BFCL cached evidence packet."""

from collections import Counter
from hashlib import file_digest
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
GROUPS = (
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)
STAGE_SIZES = {"smoke": 4, "early": 16, "preview": 40, "full": 160}
ALPHA = 0.05
MARGIN = 0.05
REQUEST_KEYS = {
    "cache_n",
    "cache_prompt",
    "model_wire_s",
    "predicted_ms",
    "predicted_n",
    "prompt_ms",
    "prompt_n",
    "request",
}
LEAK_PATTERNS = re.compile(r"/(?:home|media)/|" + "G" + "PU-" + r"|127\.0\.0\.1|" + "nm" + "moon-")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def binomial_cdf(k, n, p):
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def upper_loss_bound(losses, n, alpha=ALPHA):
    if losses >= n:
        return 1.0
    left, right = (losses + 1) / n, 1.0
    for _ in range(60):
        middle = (left + right) / 2.0
        if binomial_cdf(losses, n, middle) > alpha:
            left = middle
        else:
            right = middle
    return right


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


def verify_physical(physical, mode):
    require(physical["mode"] == mode, f"Physical mode mismatch: {physical['mode']}")
    compact = physical["compact"]
    guard = physical["compact_guard"]
    counters = physical["physical_counters"]
    require(min(compact.values()) >= 0 and min(guard.values()) >= 0 and min(counters.values()) >= 0,
            f"Negative counter in {mode}")
    require(compact["requested"] >= compact["compact_only"],
            f"Invalid compact_only in {mode}")
    offers = sum(guard.values()) - compact["compact_only"]
    require(offers >= 0 and physical["observed_fallback_proposals"] == offers,
            f"Fallback proposals mismatch in {mode}")
    require(physical["down4_launches"] == counters["down4"],
            f"down4_launches mismatch in {mode}")
    require(physical["down4_matches_40_times_guard_offers"] == (counters["down4"] == 40 * sum(guard.values())),
            f"down4_matches_40_times_guard_offers mismatch in {mode}")


def verify_scores(score):
    stage = score["stage"]
    require(stage in STAGE_SIZES, f"Unsupported stage: {stage}")
    expected_total = STAGE_SIZES[stage]
    expected_per_stratum = expected_total // len(GROUPS)
    require(all(v is True for v in score["same_build_checks"].values()),
            "same_build_checks failed")

    by_arm = {}
    for mode in ("serial", "native"):
        arm = score["arms"][mode]
        cases = arm["cases"]
        ids = [c["id"] for c in cases]
        require(len(cases) == len(set(ids)) == expected_total,
                f"{mode} case count or uniqueness check failed")
        cats = Counter(c["category"] for c in cases)
        require(cats == {g: expected_per_stratum for g in GROUPS},
                f"{mode} stratum balance failed: {dict(cats)}")
        require(sum(1 for c in cases if c["passed"]) == arm["pass_count"], f"{mode} pass_count mismatch")
        require(sum(1 for c in cases if c["status"] == "scored") == arm["scored_cases"], f"{mode} scored_cases mismatch")
        require(sum(1 for c in cases if c["model_cap"]) == arm["model_cap_cases"], f"{mode} model_cap_cases mismatch")
        require(sum(1 for c in cases if c["infra_error"]) == arm["infra_error_cases"], f"{mode} infra_error_cases mismatch")
        require(sum(c["request_count"] for c in cases) == arm["request_count"], f"{mode} request_count mismatch")
        require(sum(c["model_turns"] for c in cases) == arm["model_turns"], f"{mode} model_turns mismatch")
        require(sum(c["assistant_tool_calls"] for c in cases) == arm["assistant_tool_calls"], f"{mode} tool_calls mismatch")
        require(sum(c["tool_messages"] for c in cases) == arm["tool_messages"], f"{mode} tool_messages mismatch")
        require(math.isclose(sum(c["wall_s"] for c in cases), arm["case_wall_s"], rel_tol=0, abs_tol=1e-9),
                f"{mode} case_wall_s mismatch")
        require(math.isclose(sum(c["working_s"] for c in cases), arm["case_working_s"], rel_tol=0, abs_tol=1e-9),
                f"{mode} case_working_s mismatch")
        for c in cases:
            if not c["infra_error"] and c["limit"] in (None, "token", "message", "turn"):
                require(c["request_count"] == c["model_turns"] + int(c["limit"] == "token"),
                        f"{mode} {c['id']} request_count vs model_turns mismatch")
        obs_status = arm.get("observation_status", "nominal")
        if obs_status == "aborted_mid_run_batch_overflow":
            require(arm["physical"] is None and "abort_diagnostic" in arm,
                    f"Aborted {mode} must have physical=None and abort_diagnostic")
        else:
            verify_physical(arm["physical"], mode)
        by_arm[mode] = {c["id"]: c for c in cases}

    require(set(by_arm["serial"]) == set(by_arm["native"]), "Serial and native ID sets differ")

    names = ("both_pass", "both_fail", "on_only", "off_only", "off_infra", "on_infra",
             "off_missing", "on_missing", "off_model_cap", "on_model_cap", "unresolved_pairs")
    counts = {g: Counter({n: 0 for n in names}) for g in GROUPS}
    for cid, off in by_arm["serial"].items():
        on = by_arm["native"][cid]
        require(off["category"] == on["category"], f"Category mismatch for {cid}")
        cnt = counts[off["category"]]
        valid = True
        for label, rec in (("off", off), ("on", on)):
            if rec["status"] == "infra" and rec["score"] is None:
                cnt[f"{label}_infra"] += 1
                valid = False
            elif rec["status"] == "model_cap" and rec["score"] == 0:
                cnt[f"{label}_model_cap"] += 1
            elif rec["status"] == "scored" and rec["score"] in (0, 1):
                pass
            else:
                raise ValueError(f"Invalid sample status/score for {label} {cid}")
        if valid:
            key = {(1, 1): "both_pass", (0, 0): "both_fail", (0, 1): "on_only", (1, 0): "off_only"}[(off["score"], on["score"])]
            cnt[key] += 1
        else:
            cnt["unresolved_pairs"] += 1

    paired = score["paired_report"]
    for g in GROUPS:
        require(dict(counts[g]) == paired["groups"][g], f"Stratum {g} paired counts mismatch")
    agg = dict(sum(counts.values(), Counter()))
    require(agg == paired["total"], "Aggregate paired counts mismatch")

    unresolved = agg.get("unresolved_pairs", 0)
    losses = agg.get("off_only", 0)
    gains = agg.get("on_only", 0)
    both = agg.get("both_pass", 0)
    lower = -upper_loss_bound(losses + unresolved, expected_total)
    require(paired["planned_pairs"] == expected_total, "planned_pairs mismatch")
    require(paired["resolved_pairs"] == expected_total - unresolved, "resolved_pairs mismatch")
    require(math.isclose(paired["paired_delta_lower_95"], lower, rel_tol=0, abs_tol=1e-12),
            "paired_delta_lower_95 mismatch")
    if not unresolved:
        require(math.isclose(paired["off_accuracy"], (both + losses) / expected_total, rel_tol=0, abs_tol=1e-12),
                "off_accuracy mismatch")
        require(math.isclose(paired["on_accuracy"], (both + gains) / expected_total, rel_tol=0, abs_tol=1e-12),
                "on_accuracy mismatch")
        require(math.isclose(paired["paired_delta"], (gains - losses) / expected_total, rel_tol=0, abs_tol=1e-12),
                "paired_delta mismatch")
    else:
        require(paired["off_accuracy"] is None and paired["on_accuracy"] is None and paired["paired_delta"] is None,
                "Expected None accuracy/delta with unresolved pairs")
    expected_ni = (
        "eligible_pass"
        if stage == "full" and not unresolved and lower > -MARGIN
        and score["arms"]["serial"]["inspect_status"] == score["arms"]["native"]["inspect_status"] == "success"
        else "unresolved"
    )
    require(paired["noninferiority"] == expected_ni, "noninferiority mismatch")


def verify_requests_and_timing(score, timing, requests):
    require(set(requests.keys()) == {"serial", "native"}, "Requests must have serial and native arms")
    for mode in ("serial", "native"):
        score_cases = {c["id"]: c for c in score["arms"][mode]["cases"]}
        arm_t = timing["arms"][mode]
        arm_r = requests[mode]
        require(set(arm_r.keys()) == set(score_cases.keys()), f"{mode} request IDs differ from score IDs")
        t_cases = {c["id"]: c for c in arm_t["cases"]}
        require(set(t_cases.keys()) == set(score_cases.keys()), f"{mode} timing IDs differ from score IDs")

        sum_req = 0
        sum_cache_n = 0
        sum_prompt_n = 0
        sum_pred_n = 0
        sum_prompt_ms = 0.0
        sum_pred_ms = 0.0
        sum_wire_s = 0.0
        sum_wall_s = 0.0
        sum_work_s = 0.0
        max_wall = 0.0

        for cid, rows in arm_r.items():
            prev_total = None
            c_cache_n = 0
            c_prompt_n = 0
            c_pred_n = 0
            c_prompt_ms = 0.0
            c_pred_ms = 0.0
            c_wire_s = 0.0
            for idx, row in enumerate(rows, start=1):
                require(set(row.keys()) == REQUEST_KEYS, f"Unexpected request keys in {mode} {cid}")
                require(row["request"] == idx, f"Non-contiguous request index in {mode} {cid}")
                if idx == 1:
                    require(row["cache_prompt"] is False and row["cache_n"] == 0,
                            f"Cold first-request invariant violated in {mode} {cid}")
                else:
                    require(row["cache_prompt"] is True and 0 <= row["cache_n"] <= prev_total,
                            f"Prefix cache bound violated in {mode} {cid} req {idx}")
                require(type(row["prompt_n"]) is int and row["prompt_n"] >= 1, f"Invalid prompt_n in {mode} {cid}")
                require(type(row["cache_n"]) is int and row["cache_n"] >= 0, f"Invalid cache_n in {mode} {cid}")
                require(type(row["predicted_n"]) is int and row["predicted_n"] >= 0, f"Invalid predicted_n in {mode} {cid}")
                for fkey in ("prompt_ms", "predicted_ms", "model_wire_s"):
                    require(type(row[fkey]) in (int, float) and math.isfinite(row[fkey]) and row[fkey] >= 0,
                            f"Invalid {fkey} in {mode} {cid}")
                prev_total = row["cache_n"] + row["prompt_n"] + row["predicted_n"]
                c_cache_n += row["cache_n"]
                c_prompt_n += row["prompt_n"]
                c_pred_n += row["predicted_n"]
                c_prompt_ms += row["prompt_ms"]
                c_pred_ms += row["predicted_ms"]
                c_wire_s += row["model_wire_s"]

            tc = t_cases[cid]
            sc = score_cases[cid]
            require(len(rows) == tc["request_count"] == sc["request_count"],
                    f"Request count mismatch for {mode} {cid}")
            require(c_cache_n == tc["cache_n"] and c_prompt_n == tc["prompt_n"] and c_pred_n == tc["predicted_n"],
                    f"Token totals mismatch for {mode} {cid}")
            require(math.isclose(c_prompt_ms, tc["prompt_ms"], rel_tol=0, abs_tol=1e-6),
                    f"prompt_ms mismatch for {mode} {cid}")
            require(math.isclose(c_pred_ms, tc["predicted_ms"], rel_tol=0, abs_tol=1e-6),
                    f"predicted_ms mismatch for {mode} {cid}")
            require(math.isclose(c_wire_s, tc["model_wire_s"], rel_tol=0, abs_tol=1e-9),
                    f"model_wire_s mismatch for {mode} {cid}")
            require(math.isclose(tc["wall_s"], sc["wall_s"], rel_tol=0, abs_tol=1e-9),
                    f"wall_s mismatch for {mode} {cid}")
            sum_req += len(rows)
            sum_cache_n += c_cache_n
            sum_prompt_n += c_prompt_n
            sum_pred_n += c_pred_n
            sum_prompt_ms += c_prompt_ms
            sum_pred_ms += c_pred_ms
            sum_wire_s += c_wire_s
            sum_wall_s += tc["wall_s"]
            sum_work_s += tc["working_s"]
            if tc["wall_s"] > max_wall:
                max_wall = tc["wall_s"]

        tot = arm_t["totals"]
        require(sum_req == tot["request_count"], f"{mode} total request_count mismatch")
        require(sum_cache_n == tot["cache_n"] and sum_prompt_n == tot["prompt_n"] and sum_pred_n == tot["predicted_n"],
                f"{mode} total token mismatch")
        require(math.isclose(sum_prompt_ms, tot["prompt_ms"], rel_tol=0, abs_tol=1e-6), f"{mode} total prompt_ms mismatch")
        require(math.isclose(sum_pred_ms, tot["predicted_ms"], rel_tol=0, abs_tol=1e-6), f"{mode} total predicted_ms mismatch")
        require(math.isclose(sum_wire_s, tot["model_wire_s"], rel_tol=0, abs_tol=1e-9), f"{mode} total model_wire_s mismatch")
        require(math.isclose(sum_wall_s, arm_t["case_wall_s_sum"], rel_tol=0, abs_tol=1e-9), f"{mode} case_wall_s_sum mismatch")
        require(math.isclose(sum_work_s, arm_t["case_working_s_sum"], rel_tol=0, abs_tol=1e-9), f"{mode} case_working_s_sum mismatch")
        require(math.isclose(max_wall, arm_t["max_case_wall_s"], rel_tol=0, abs_tol=1e-9), f"{mode} max_case_wall_s mismatch")


def verify_launch_outcomes(score, launch):
    for mode in ("serial", "native"):
        entry = launch[mode]
        arm = score["arms"][mode]
        obs_status = arm.get("observation_status", "nominal")
        require(obs_status in ("nominal", "revalidated_after_log_rotation", "aborted_mid_run_batch_overflow"),
                f"Unexpected observation_status for {mode}: {obs_status}")
        require(entry.get("observation_status") == obs_status,
                f"Launch observation_status mismatch for {mode}")
        require(entry["completed_cases"] == arm["planned_cases"] - arm["infra_error_cases"],
                f"Launch completed_cases mismatch for {mode}")
        require(entry["wire_requests"] == arm["request_count"],
                f"Launch wire_requests mismatch for {mode}")
        require(entry["cleanup_exit"] == arm["cleanup_exit"] == 0, f"Launch cleanup_exit mismatch for {mode}")
        require(entry["evaluator_exit"] == arm["evaluator_exit"] == 0, f"Launch evaluator_exit mismatch for {mode}")
        if obs_status == "revalidated_after_log_rotation":
            require(entry["runner_exit"] == arm["runner_exit"] in (0, 1),
                    f"Launch runner_exit mismatch for revalidated {mode}")
            require("observation_revalidation" in arm, f"Missing observation_revalidation for {mode}")
        elif obs_status == "aborted_mid_run_batch_overflow":
            require(entry["runner_exit"] == arm["runner_exit"] in (0, 1),
                    f"Launch runner_exit mismatch for aborted {mode}")
            require("abort_diagnostic" in arm, f"Missing abort_diagnostic for {mode}")
        else:
            require(entry["runner_exit"] == arm["runner_exit"] == 0, f"Launch runner_exit mismatch for {mode}")


def main():
    verify_sha256sums()
    verify_leaks()
    score = read_json("evidence/bfcl-full-score.json")
    timing = read_json("evidence/bfcl-full-timing.json")
    requests = read_json("evidence/bfcl-full-requests.json")
    launch = read_json("evidence/launch-outcomes.json")
    verify_scores(score)
    verify_requests_and_timing(score, timing, requests)
    verify_launch_outcomes(score, launch)
    print(f"Verified {score['stage']} BFCL cached pair ({score['paired_report']['planned_pairs']} cases), "
          f"request totals, physical counters, launch disclosures, and SHA256SUMS.")


if __name__ == "__main__":
    main()
