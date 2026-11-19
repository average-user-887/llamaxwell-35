"""Verify the public two-pair smoke summary using the standard library."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
report = json.loads((root / "evidence/agentic-local-r2-smoke-pair.json").read_text())
travel, retail = report["cases"]
assert report["stage"] == "smoke" and report["predeclared_pairs"] == 2
assert [travel["id"], retail["id"]] == ["agentdojo-travel-u1", "tau2-retail-23"]
assert report["same_r2_build_and_budgets_verified"] and report["observation_seals_verified"]
for mode, arm in report["arms"].items():
    assert arm["runner_exit"] == arm["evaluator_exit"] == arm["cleanup_exit"] == 0
    assert arm["evaluator_wall_s"] == 148
    assert arm["routes"]["experts_used"] == 8 and arm["routes"]["gpu_layers_offloaded"] == "42/42"
    assert arm["routes"]["compact_only"] == 0
    assert arm["routes"]["down4_launches"] == 40 * arm["routes"]["sampler_guard_fallbacks"]
    assert math.isclose(arm["episode_wall_sum_s"], travel[mode]["wall_s"] + retail[mode]["wall_s"])
    assert travel[mode]["status"] == "completed" and travel[mode]["error_code"] is None
    assert travel[mode]["malformed_tool_arguments"] == travel[mode]["tool_rejections"] == 1
    assert travel[mode]["generation_length_ends"] == 1
    assert retail[mode]["budget_reason"] == "context_tokens" and retail[mode]["error_code"] is None
    assert retail[mode]["max_prompt_tokens"] > report["logical_context_cap_tokens"] == 8192
    assert retail[mode]["tool_output_truncations"] == 1
assert report["arms"]["native"]["routes"]["sampler_guard_fallbacks"] == 728
assert report["arms"]["serial"]["routes"]["sampler_guard_fallbacks"] == 0
for metric in ("state", "joint"):
    pairs = [(case["serial"][metric], case["native"][metric]) for case in report["cases"]]
    assert pairs == [(1, 1), (0, 0)]
    assert report["scores"][metric] == {
        "serial_pass": sum(s == 1 for s, _ in pairs),
        "native_pass": sum(n == 1 for _, n in pairs),
        "native_wins": pairs.count((0, 1)),
        "native_losses": pairs.count((1, 0)),
        "ties": pairs.count((0, 0)) + pairs.count((1, 1)),
        "unknown_pairs": sum(s is None or n is None for s, n in pairs),
    }
print("Verified two matched smoke pairs, both metric summaries, run status and route counters.")
