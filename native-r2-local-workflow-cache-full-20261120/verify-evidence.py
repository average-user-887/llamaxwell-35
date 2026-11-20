#!/usr/bin/env python3
"""Standalone zero-dependency verifier for the public agentic-local-cache-r2 packet."""

from hashlib import file_digest
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
STAGE_SIZES = {"smoke": 2, "full": 80}
REQUEST_KEYS = {
    "cache_n",
    "cache_prompt",
    "elapsed_s",
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
    require(compact["requested"] >= compact["compact_only"], f"Invalid compact_only in {mode}")
    offers = sum(guard.values()) - compact["compact_only"]
    require(offers >= 0 and physical["observed_fallback_proposals"] == offers,
            f"Fallback proposals mismatch in {mode}")
    require(physical["down4_launches"] == counters["down4"], f"down4_launches mismatch in {mode}")
    require(physical["down4_matches_40_times_guard_offers"] == (counters["down4"] == 40 * sum(guard.values())),
            f"down4_matches_40_times_guard_offers mismatch in {mode}")


def verify_pair_and_requests(pair, requests, launch):
    stage = pair["stage"]
    require(stage in STAGE_SIZES, f"Unsupported stage: {stage}")
    expected_n = STAGE_SIZES[stage]
    require(pair["predeclared_pairs"] == len(pair["cases"]) == expected_n, "Case count mismatch")
    require(len({c["id"] for c in pair["cases"]}) == expected_n, "Duplicate case IDs")
    require(pair["same_r2_build_and_budgets_verified"] is True, "same_r2_build_and_budgets_verified failed")
    require(pair["observation_seals_verified"] is True, "observation_seals_verified failed")
    require(all(v is True for v in pair["same_build_checks"].values()), "same_build_checks failed")

    for mode in ("serial", "native"):
        arm = pair["arms"][mode]
        obs_status = arm.get("observation_status", "nominal")
        require(obs_status in ("nominal", "revalidated_after_log_rotation"),
                f"Invalid observation_status for {mode}")
        require(arm["cleanup_exit"] == arm["evaluator_exit"] == 0, f"Arm cleanup/evaluator failed for {mode}")
        if obs_status == "nominal":
            require(arm["runner_exit"] == 0, f"Expected runner_exit == 0 for {mode}")
        else:
            require(arm["runner_exit"] in (0, 1) and "observation_revalidation" in arm,
                    f"Invalid revalidated runner_exit for {mode}")
        verify_physical(arm["physical"], mode)

        arm_reqs = requests[mode]
        require(set(arm_reqs.keys()) == {c["id"] for c in pair["cases"]},
                f"{mode} request IDs differ from cases")

        sum_wall = 0.0
        max_wall = 0.0
        sum_fields = {k: 0 for k in (
            "chat_requests", "cache_reuse_requests", "cache_n", "prompt_n", "predicted_n",
            "prompt_tokens", "completion_tokens", "prompt_ms", "predicted_ms",
            "assistant_tool_calls", "tool_messages", "tool_rejections", "tool_output_truncations",
            "malformed_tool_arguments", "malformed_tool_calls", "generation_length_ends",
            "empty_assistant_turns",
        )}
        for entry in pair["cases"]:
            cid = entry["id"]
            c = entry[mode]
            rows = arm_reqs[cid]
            require(len(rows) == c["chat_requests"], f"{mode} {cid} chat_requests mismatch")
            prev = None
            c_cache_n = 0
            c_prompt_n = 0
            c_pred_n = 0
            c_reuse = 0
            c_pms = 0.0
            c_dms = 0.0
            for idx, r in enumerate(rows, start=1):
                require(set(r.keys()) == REQUEST_KEYS, f"Unexpected keys in {mode} {cid}")
                require(r["request"] == idx, f"Non-contiguous request index in {mode} {cid}")
                if idx == 1:
                    require(r["cache_prompt"] is False and r["cache_n"] == 0,
                            f"Cold entry reused tokens in {mode} {cid}")
                else:
                    require(r["cache_prompt"] is True and 0 <= r["cache_n"] <= prev,
                            f"Reused prefix out of bounds in {mode} {cid} req {idx}")
                require(r["prompt_n"] >= 1 and r["predicted_n"] >= 0, f"Invalid token count in {mode} {cid}")
                prev = r["cache_n"] + r["prompt_n"] + r["predicted_n"]
                c_cache_n += r["cache_n"]
                c_prompt_n += r["prompt_n"]
                c_pred_n += r["predicted_n"]
                c_reuse += int(r["cache_n"] > 0)
                c_pms += r["prompt_ms"]
                c_dms += r["predicted_ms"]

            require(c_cache_n == c["cache_n"] and c_prompt_n == c["prompt_n"] and c_pred_n == c["predicted_n"],
                    f"Episode token totals mismatch in {mode} {cid}")
            require(c_reuse == c["cache_reuse_requests"], f"cache_reuse_requests mismatch in {mode} {cid}")
            require(math.isclose(c_pms, c["prompt_ms"], rel_tol=0, abs_tol=1e-6), f"prompt_ms mismatch in {mode} {cid}")
            require(math.isclose(c_dms, c["predicted_ms"], rel_tol=0, abs_tol=1e-6), f"predicted_ms mismatch in {mode} {cid}")
            sum_wall += c["wall_s"]
            if c["wall_s"] > max_wall:
                max_wall = c["wall_s"]
            for k in sum_fields:
                sum_fields[k] += c[k]

        require(math.isclose(sum_wall, arm["episode_wall_sum_s"], rel_tol=0, abs_tol=1e-6),
                f"{mode} episode_wall_sum_s mismatch")
        require(math.isclose(max_wall, arm["max_episode_wall_s"], rel_tol=0, abs_tol=1e-6),
                f"{mode} max_episode_wall_s mismatch")
        for k, v in sum_fields.items():
            if isinstance(v, float):
                require(math.isclose(v, arm["totals"][k], rel_tol=0, abs_tol=1e-6), f"{mode} total {k} mismatch")
            else:
                require(v == arm["totals"][k], f"{mode} total {k} mismatch")

        l_entry = launch[mode]
        require(l_entry["observation_status"] == obs_status, f"Launch observation_status mismatch for {mode}")
        require(l_entry["wire_requests"] == arm["totals"]["chat_requests"], f"Launch wire_requests mismatch for {mode}")
        require(l_entry["runner_exit"] == arm["runner_exit"], f"Launch runner_exit mismatch for {mode}")

    for metric in ("state", "joint"):
        pairs = [(c["serial"][metric], c["native"][metric]) for c in pair["cases"]]
        expected_score = {
            "serial_pass": sum(s == 1 for s, _ in pairs),
            "native_pass": sum(n == 1 for _, n in pairs),
            "native_wins": pairs.count((0, 1)),
            "native_losses": pairs.count((1, 0)),
            "ties": pairs.count((0, 0)) + pairs.count((1, 1)),
            "unknown_pairs": sum(s is None or n is None for s, n in pairs),
        }
        require(pair["scores"][metric] == expected_score, f"Score tally mismatch for {metric}")


def main():
    verify_sha256sums()
    verify_leaks()
    pair = read_json("evidence/agentic-local-cache-pair.json")
    requests = read_json("evidence/agentic-local-cache-requests.json")
    launch = read_json("evidence/launch-outcomes.json")
    verify_pair_and_requests(pair, requests, launch)
    print(f"Verified {pair['stage']} agentic-local-cache-r2 pair ({pair['predeclared_pairs']} cases), "
          f"wire prefix reuse, physical counters, and SHA256SUMS.")


if __name__ == "__main__":
    main()
