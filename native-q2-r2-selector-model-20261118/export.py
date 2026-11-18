"""Export typed observations from completed local receipts, keeping model output private."""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CARD = HERE.parents[1] / "last-ms-r2-selector-model"
PHASES = ("cold", "warm")
COST = re.compile(r"\b(draft|tgt_decode|spec_process|sample_accept)_ms=([0-9.]+)\(n=(\d+)\)")
SNAPSHOT = re.compile(r"^mv1_host: rounds=(\d+) restores=(\d+) (.*) acc_hist=\[(\d+),(\d+),(\d+),(\d+)\]$", re.M)
STATUS = re.compile(r"^(runner_exit|cleanup_exit|completed_epoch|arm_deadline_epoch)=(\d+)$", re.M)

def read(path):
    return json.loads(path.read_text())

def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def repetition(text):
    clean = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    lines = [re.sub(r"\d+", "#", line) for line in clean]
    tail = lines[3*len(lines)//4:]
    words = re.findall(r"\w+", re.sub(r"\d+", "#", " ".join(text.split())))
    windows = [tuple(words[i:i+24]) for i in range(max(0, len(words)-23))]
    lead = re.compile(r"^(?:(?:async\s+)?(?:def|for|with)|class|if|elif|else|while|try|except|finally|return|raise|yield|from|import|match|case)\b")
    return {
        "nonblank_lines": len(lines), "duplicate_normalized_lines": sum(n-1 for n in Counter(lines).values()),
        "final_quarter_lines": len(tail), "final_quarter_normalized_duplicates": sum(n-1 for n in Counter(tail).values()),
        "word_windows_24": len(windows), "repeated_24_word_windows": sum(n-1 for n in Counter(windows).values()),
        "python_keyword_led_lines": sum(bool(lead.match(line)) for line in clean),
        "trailing_comment_only_lines": next((i for i, line in enumerate(reversed(clean)) if not line.startswith("#")), len(clean)),
        "malformed_characters": sum(ord(c) == 0xfffd or (ord(c) < 32 and c not in "\r\t\n") for c in text),
    }

def parity(a, b):
    ax, bx = a["token_ids"], b["token_ids"]
    first = next((i for i, (u, v) in enumerate(zip(ax, bx), 1) if u != v),
                 min(len(ax), len(bx)) + 1 if len(ax) != len(bx) else None)
    return {
        "left_ids": len(ax), "right_ids": len(bx), "first_id_difference_1_based": first,
        "full_text_equal": a["text"] == b["text"], "repetition_metrics_equal": repetition(a["text"]) == repetition(b["text"]),
        "same_proposal_and_acceptance_totals": all(a["timings"].get(k) == b["timings"].get(k)
                                                for k in ("draft_n", "draft_n_accepted")),
    }

def host_points(receipt):
    final = (receipt / "server.log").read_text()
    prewarm = (receipt / "server-pre-warm.log").read_text()
    assert final.startswith(prewarm)
    parsed = {}
    for rounds, restores, costs, *hist in SNAPSHOT.findall(final[len(prewarm):]):
        index = int(rounds)
        assert index not in parsed
        parsed[index] = (int(restores), tuple(map(int, hist)),
                         {k: (float(ms), int(count)) for k, ms, count in COST.findall(costs)})
    return parsed

def host_comparison(earlier, current):
    old, new = host_points(earlier), host_points(current)
    matched = sorted(old.keys() & new.keys())
    if len(matched) < 2:
        return {"matched_snapshots": len(matched), "window_available": False}
    assert all(old[i][:2] == new[i][:2] for i in matched)
    start, end = matched[0], matched[-1]
    span = end - start
    costs = {}
    for key in ("draft", "tgt_decode", "spec_process", "sample_accept"):
        old_ms, old_calls = [v - u for u, v in zip(old[start][2][key], old[end][2][key])]
        new_ms, new_calls = [v - u for u, v in zip(new[start][2][key], new[end][2][key])]
        costs[key] = {"prior_total_ms": old_ms, "candidate_total_ms": new_ms,
                      "saved_ms_per_offered_round": (old_ms - new_ms) / span,
                      "prior_calls": old_calls, "candidate_calls": new_calls}
    return {"matched_snapshots": len(matched), "window_available": True, "first_round": start, "last_round": end,
            "matched_rounds": span, "matched_cumulative_acceptance_histograms": True, "host_components": costs,
            "scope": "Cumulative host phase time covers all work in that phase. It has no exclusive CUDA kernel attribution."}

def receipt(out):
    status = dict(STATUS.findall((out / "status.txt").read_text()))
    official = read(out / "sanitized/model-receipt.json")
    assert "completed_epoch" in status and all(int(status[key]) == official[key] for key in ("runner_exit", "cleanup_exit"))
    response = {}
    for phase in PHASES:
        path = out / f"{phase}.json"
        if path.is_file():
            result = read(path)
            timing = result["timings"]
            response[phase] = {"output_count": len(result["token_ids"]), "finish_reason": result["finish_reason"],
                               "request_wall_s": result["wall_s"], "ttft_s": result["ttft_s"],
                               "server_prompt_ms": timing.get("prompt_ms"), "server_decode_ms": timing.get("predicted_ms"),
                               "offered_drafts": timing.get("draft_n"), "accepted_drafts": timing.get("draft_n_accepted"),
                               "repetition": repetition(result["text"])}
    validated = official["validated"]
    if validated is not None:
        allowed = ("mode", "technical_pass", "semantic_score", "actual_server_software_sha256", "metrics",
                   "phase_counter_evidence", "accepted_experts", "outputs_vs_archived_q8_native", "native_routes",
                   "observed_q2_residency", "serial_sparse_widths", "outputs_vs_previous_same_mode_19607daf",
                   "selector_execution_witness")
        validated = {key: validated[key] for key in allowed}
    return {"mode": official["mode"], "runner_exit": official["runner_exit"], "cleanup_exit": official["cleanup_exit"],
            "bounded_wall_s": official["bounded_wall_s"], "refusal": official["refusal"], "request": response,
            "read_only_counter_snapshots": official["snapshots"], "validated": validated}

def private_model():
    spec = importlib.util.spec_from_file_location("selector_model", CARD / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def verify_observations(path):
    subprocess.run(["sha256sum", "--check", "--quiet", "observations.sha256"], cwd=path, check=True)

def audit(serial, native, model):
    arms = {"current_native": native, "current_serial": serial, "prior_19607daf_native": model.REFERENCE["native"],
            "prior_19607daf_serial": model.REFERENCE["serial"], "earlier_q8_native": model.prior.REF}
    report = {}
    for phase in PHASES:
        records = {key: read(path / f"{phase}.json") for key, path in arms.items()}
        rows = {key: repetition(value["text"]) for key, value in records.items()}
        comparisons = {
            "current_native_vs_current_serial": parity(records["current_native"], records["current_serial"]),
            "current_native_vs_prior_19607daf_native": parity(records["current_native"], records["prior_19607daf_native"]),
            "current_native_vs_earlier_q8_native": parity(records["current_native"], records["earlier_q8_native"]),
            "current_serial_vs_prior_19607daf_serial": parity(records["current_serial"], records["prior_19607daf_serial"]),
        }
        for name, checks in comparisons.items():
            if name == "current_native_vs_current_serial":
                continue
            assert checks["first_id_difference_1_based"] is None and checks["full_text_equal"]
            assert checks["repetition_metrics_equal"] and checks["same_proposal_and_acceptance_totals"]
        report[phase] = {"repetition": rows, "comparisons": comparisons}
    return {"normalization": "collapse whitespace and replace digit runs; count excess occurrences",
            "semantic_score": None, "phases": report}

if __name__ == "__main__":
    args = sys.argv[1:]
    mod = private_model()
    if args == ["--offline"]:
        staged = audit(mod.REFERENCE["serial"], mod.REFERENCE["native"], mod)
        warm = staged["phases"]["warm"]
        assert warm["comparisons"]["current_native_vs_current_serial"]["first_id_difference_1_based"] == 170
        assert warm["repetition"]["current_native"]["duplicate_normalized_lines"] == 127
        assert warm["repetition"]["current_serial"]["repeated_24_word_windows"] == 46
        assert host_comparison(mod.REFERENCE["native"], mod.REFERENCE["native"])["matched_rounds"] == 1618
        print("Independent repetition and host-window audit validated against prior native and strict serial references.")
    elif len(args) == 2:
        mode, path = args[0], Path(args[1]).resolve()
        assert mode in ("native", "serial")
        verify_observations(path)
        data = receipt(path)
        assert data["mode"] == mode
        write(f"{mode}.receipt.json", data)
    elif len(args) == 3 and args[0] == "--compare":
        serial, native = map(lambda p: Path(p).resolve(), args[1:])
        for path in (serial, native):
            verify_observations(path)
        write("comparison.json", mod.compare(serial, native))
        write("repetition.json", audit(serial, native, mod))
        write("host-native-comparison.json", host_comparison(mod.REFERENCE["native"], native))
    else:
        raise SystemExit("usage: export.py --offline | native|serial RECEIPT | --compare SERIAL NATIVE")
