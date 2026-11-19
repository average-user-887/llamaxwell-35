"""Build an anonymous receipt from the isolated target observer."""
import json
from pathlib import Path
import re
import sys

EXPECTED_GRAPHS = {
    "accepted": (2, 80167), "serial_first": (1, 80167), "serial_second": (1, 80168),
    "wrong_a": (2, 80167), "correct_a": (1, 80168), "wrong_b": (2, 80167), "correct_b": (1, 80168),
}
PERMITTED = {
    "PREFIX": {"positions", "bytes", "offload", "rs", "outputs"},
    "RESTORE": {"bitwise"},
    "GRAPH": {"input", "output", "calls", "position", "sparse", "invalid_sparse", "padded_kv", "t2",
              "t2_attempts", "t2_fallback", "moe"},
    "ACCEPTED": {"row0_matches_archived_accepted_token"},
    "STATE": {"kv", "conv", "ssm", "a_prefix", "b_prefix"},
    "SCORE": {"raw", "greedy", "top2_gap", "serial_logp", "native_logp", "serial_minus_native"},
    "LOGITS": {"different_f32_words", "max_abs"},
    "TRIM": {"removed", "genuine", "retained_position"},
    "COMPLETE": {"width_flip_matches_archived", "genuine_two_rejections"},
}
VALUE = re.compile(r"(?:-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|true|false|serial|native|other|42/42|\d+,\d+,\d+,\d+)")


def parse(log):
    records = {tag: {} for tag in PERMITTED}
    for line in log.splitlines():
        head, _, body = line.partition(" ")
        if head not in PERMITTED:
            continue
        fields = body.split()
        key = "" if "=" in fields[0] else fields.pop(0)
        if key and not re.fullmatch(r"[a-z0-9_]+", key):
            raise ValueError("invalid receipt record name")
        values = {k: v for k, v in (field.split("=", 1) for field in fields)}
        if (key in records[head] or set(values) != PERMITTED[head] or
                any(not VALUE.fullmatch(v) for v in values.values())):
            raise ValueError("invalid, sensitive or duplicated receipt record")
        records[head][key] = values
    return records


def summarize(log, runner_exit, cleanup_exit):
    records = parse(log)
    graphs = records["GRAPH"]
    def graph_matches(name):
        if name not in graphs:
            return False
        width, pos = EXPECTED_GRAPHS[name]
        g = graphs[name]
        return (int(g["input"]) == int(g["output"]) == width and int(g["position"]) == pos and
                g["calls"] == "1" and g["sparse"] == "10" and g["invalid_sparse"] == "0" and
                g["padded_kv"] == "80384" and g["t2"] == g["t2_attempts"] == ("30" if width == 2 else "0") and
                g["t2_fallback"] == "0" and g["moe"] == ("0,0,40,40" if width == 2 else "0,0,0,0"))
    graph_witness = set(graphs) == set(EXPECTED_GRAPHS) and all(map(graph_matches, EXPECTED_GRAPHS))
    stride = bool(re.search(r"q2_r2_stride_dispatch width=80384 k_token_bytes=1024 v_token_bytes=1024 ", log))
    top16 = "q2_r2_top16_dispatch n_q=2 n_kv=80384 nkvh=2 plans=4" in log
    q8 = "N82_ADMISSION original_blocks=3 candidate_blocks=3 grid=72 admitted=1" in log
    runtime_maps = sum(line.startswith("QUALITY80_MAPPED ") for line in log.splitlines()) == 4
    experts = bool(re.search(r"\bn_expert_used\s+= 8\b", log))
    expected_states = {"accepted_vs_serial", "wrong_untrimmed_a_vs_b", "wrong_retained_a_vs_b",
                       "wrong_corrected_a_vs_b", "wrong_retained_a_vs_serial", "wrong_corrected_a_vs_serial"}
    expected_logits = {"accepted_row1_vs_serial", "wrong_row0_a_vs_b", "wrong_row1_a_vs_b",
                       "wrong_row0_a_vs_accepted_row0", "wrong_corrected_a_vs_b", "wrong_corrected_a_vs_serial"}
    expected_scores = {"accepted_row1", "serial_row1_conditioning", "correct_a", "correct_b"}
    state = records["STATE"]
    logits = records["LOGITS"]
    active_prefix = bool(state) and all(v["a_prefix"] == v["b_prefix"] == "true" for v in state.values())
    complete = records["COMPLETE"].get("")
    executed = (runner_exit == cleanup_exit == 0 and graph_witness and stride and top16 and q8 and experts and runtime_maps and
                complete is not None and set(state) == expected_states and set(logits) == expected_logits and
                set(records["SCORE"]) == expected_scores and
                set(records["TRIM"]) == {"wrong_a", "wrong_b"} and all(
                    v["removed"] == "1" and v["retained_position"] == "80167" for v in records["TRIM"].values()) and
                set(records["RESTORE"]) == {"accepted", "serial", "wrong_a", "wrong_b"} and
                all(values["bitwise"] == "1" for values in records["RESTORE"].values()))
    same_shape = None
    if executed and complete["genuine_two_rejections"] == "true":
        def equal(stage):
            return all(state[stage][c] == "true" for c in ("kv", "conv", "ssm"))
        same_shape = (active_prefix and not equal("wrong_untrimmed_a_vs_b") and
                      int(logits["wrong_row1_a_vs_b"]["different_f32_words"]) > 0 and
                      equal("wrong_retained_a_vs_b") and equal("wrong_corrected_a_vs_b") and
                      logits["wrong_row0_a_vs_b"]["different_f32_words"] == "0" and
                      logits["wrong_corrected_a_vs_b"]["different_f32_words"] == "0")
    scores = records["SCORE"]
    width_ready = (
        cleanup_exit == 0 and stride and top16 and q8 and experts and runtime_maps and
        all(map(graph_matches, ("accepted", "serial_first", "serial_second"))) and
        all(records["RESTORE"].get(arm) == {"bitwise": "1"} for arm in ("accepted", "serial")) and
        "accepted_vs_serial" in state and "accepted_row1_vs_serial" in logits and
        {"accepted_row1", "serial_row1_conditioning"} <= set(scores)
    )
    width_flip = None if not width_ready else (
        scores["accepted_row1"]["greedy"] == "native" and scores["serial_row1_conditioning"]["greedy"] == "serial"
    )
    matches_live_margins = None if not width_ready else all(
        abs(float(scores[stage]["serial_minus_native"]) - expected) <= 0.00002
        for stage, expected in [("accepted_row1", -0.53683), ("serial_row1_conditioning", 0.66038)]
    )
    return {
        "matches_logged_live_pair_margins_within_2e_5": matches_live_margins,
        "runner_exit": runner_exit, "cleanup_exit": cleanup_exit, "executed_required_paths": executed,
        "witness": {"graphs": graph_witness, "long_q2_stride": stride, "long_q2_top16": top16,
                    "q8_gdn_occupancy": q8, "eight_experts": experts, "runtime_maps": runtime_maps},
        "executed_width_only_paths": width_ready,
        "width_only_choice_reproduction": width_flip,
        "both_proposals_genuinely_rejected": None if not executed else complete["genuine_two_rejections"] == "true",
        "prior_active_kv_matches_common_prefix": active_prefix,
        "same_shape_suffix_isolation": same_shape,
        "records": records,
    }


if __name__ == "__main__":
    path, run, cleanup = sys.argv[1:]
    result = summarize(Path(path).read_text(errors="replace"), int(run), int(cleanup))
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["executed_required_paths"] else 2)
