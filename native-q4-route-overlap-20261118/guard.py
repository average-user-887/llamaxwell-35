#!/usr/bin/env python3
"""Apply the frozen attention-only profile and admit overlap only after exact parity."""
import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ["Q4_COMBINED_CARD"])
REFERENCE = Path(os.environ["Q4_ROUTE_PRIVATE_REFERENCE"])
MODE = "depth3-attn"
BASE_BUFFER = HERE.parent / "night-cache-only/libs/libggml-base.so.0.26.0"
MAGIC = 0x5152344f564c31
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("overlap_base_native_probe", BASE / "native_probe.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
n9 = base.n9


def __getattr__(name):
    return getattr(base, name)


def mapped(out):
    expected = {path: digest for digest, path in
                (row.split("  ", 1) for row in (REFERENCE / "runtime-library-hashes.txt").read_text().splitlines())}
    expected["/audit.so"] = n9.hash_file(HERE / "audit.so")
    expected[str(BASE_BUFFER)] = n9.hash_file(BASE_BUFFER)
    rows = [row.split("  ", 1) for row in (out / "runtime-library-hashes.txt").read_text().splitlines()]
    observed = {path: digest for digest, path in rows}
    if len(rows) != len(expected) or expected != observed:
        raise ValueError("runtime binaries and actual observer do not match the pinned route diagnostic")
    return observed


def qualify():
    base.qualify()
    base.check_moe()
    status = n9.key_values(REFERENCE / "status.txt")
    if (status["runner_exit"], status["cleanup_exit"]) != ("0", "0"):
        raise ValueError("attention-only reference failed original runtime and teardown checks")
    prior = base.collect(MODE, False, REFERENCE)
    if json.loads(json.dumps(prior)) != n9.read_json(REFERENCE / "result.json"):
        raise ValueError("pinned attention-only reference failed existing replay")


def shaped(path):
    blob = path.read_bytes()
    if len(blob) != 16384:
        raise ValueError("anonymous route counter buffer shape differs")
    words = struct.unpack("<2048Q", blob)
    if words[:2] != (MAGIC, 1):
        raise ValueError("route observer is not attached to the actual owned server")
    return words


def arm(path):
    words = shaped(path)
    if words[2] or words[3] != 1 or any(words[4:]):
        raise ValueError("expected a single target-init hook and untouched cold route counters")
    with path.open("r+b") as file:
        file.seek(16)
        file.write(struct.pack("<Q", 1))
        file.flush()


def parity(raw):
    results = {}
    for phase in ("cold", "warm"):
        reference = n9.read_json(REFERENCE / f"{phase}.json")
        actual = n9.read_json(raw / f"{phase}.json")
        keys = ("prompt_sha256", "prompt_n", "id_slot", "temperature", "seed", "finish_reason", "n_out",
                "cache_prompt", "ignore_eos", "truncated")
        if any(actual[key] != reference[key] for key in keys):
            raise ValueError("reference and diagnostic input or output contract differs")
        count = 192 if phase == "cold" else 3072
        if len(actual["token_ids"]) != count or len(reference["token_ids"]) != count:
            raise ValueError("exact output length is unavailable")
        mismatch = n9.first_mismatch(reference["token_ids"], actual["token_ids"])
        text_equal = actual["text"] == reference["text"]
        results[phase] = {"all_ids_equal": mismatch is None, "first_id_difference_1_based": mismatch,
                          "all_text_equal": text_equal, "checked_ids": count}
    return results


def aggregate(path):
    words = shaped(path)
    armed, init, eligible, selected, complete, hits, errors, calls, clears = words[2:11]
    if (armed, init) != (1, 1) or not 16 <= selected <= 32 or eligible < 16 * (selected - 1) + 1:
        raise ValueError("warm sample eligibility or requested coverage differs")
    if errors or complete != selected or hits != 40*complete or calls != selected or clears != calls or any(words[11:16]):
        raise ValueError("each sample must see exactly one synchronized graph and all 40 valid distinct layers")
    rows = []
    for layer in range(40):
        start = 16 + 29 * layer
        histogram, multiplicity = words[start:start+25], words[start+25:start+29]
        sum_union = sum((i+8)*value for i, value in enumerate(histogram))
        if sum(histogram) != complete or sum(multiplicity) != sum_union or sum(
                (i+1)*value for i, value in enumerate(multiplicity)) != 32*complete:
            raise ValueError("union and multiplicity counts do not reconcile with exact top eight")
        rows.append({"layer": layer, "window_count": complete, "expert_union_histogram_8_to_32": list(histogram),
                     "expert_multiplicity_histogram_1_to_4": list(multiplicity), "mean_expert_union": sum_union/complete})
    if any(words[16 + 29*40:]):
        raise ValueError("nonzero unexpected data in anonymous counter buffer")
    mean_union = sum(row["mean_expert_union"] for row in rows)/len(rows)
    return {"eligible_warm_q4_calls": eligible, "sample_stride": 16, "selected_q4_calls": selected,
            "complete_q4_calls": complete, "layer_samples": hits, "selected_graph_calls": calls,
            "mean_expert_union_all_layers": mean_union,
            "mean_repeated_expert_occurrences_all_layers": 32 - mean_union,
            "layers": rows}


def observe(raw):
    result = base.collect(MODE, False, raw)
    n9.write_json(raw / "result.json", result)
    comparison = parity(raw)
    n9.write_json(raw / "parity.json", comparison)
    if not all(row["all_ids_equal"] and row["all_text_equal"] for row in comparison.values()):
        raise ValueError("observer output diverged from pinned attention-only reference")
    routes = aggregate(raw / "route-live.bin")
    if routes["eligible_warm_q4_calls"] != result["metrics"]["warm"]["moe_dispatch"]["width4_target_generation_calls"]:
        raise ValueError("callback eligibility and independent main-target width-four observer disagree")
    n9.write_json(raw / "overlap.json", routes)


def export(out, runner, cleanup, started, finished):
    base.finalize(MODE, False, out, runner, cleanup, started, finished)
    public = out / "sanitized"
    raw = out / "private"
    receipt_path = public / "receipt.json"
    receipt = n9.read_json(receipt_path)
    receipt["diagnostic_observation_only"] = True
    receipt["profile_instrumented"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    success = runner == cleanup == 0 and (raw / "overlap.json").is_file()
    if runner == cleanup == 0 and not success:
        raise ValueError("successful runner lacks verified route result")
    comparison = base.partial_record(raw / "parity.json")
    result = {"scope": "anonymous top-eight route distribution of exact attention-only proxy",
              "applies_to_fused_moe_cohorts": False, "performance_measurement": False,
              "runner_exit": runner, "cleanup_exit": cleanup, "route_pass": success, "parity": comparison or None,
              "anonymous_overlap": n9.read_json(raw / "overlap.json") if success else None}
    n9.write_json(public / "route-receipt.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "args", "env", "ready", "arm", "verify", "finalize"))
    parser.add_argument("params", nargs="*")
    args = parser.parse_args()
    if args.action == "check":
        qualify()
        print("Pinned attention-only runtime and independently complete reference replay verified")
        return
    base.HERE = HERE
    base.mapped = mapped
    values = args.params
    if args.action in ("args", "env", "ready", "verify", "finalize") and values[:2] != [MODE, "0"]:
        raise ValueError("route observer only accepts the fixed attention-only mode")
    if args.action in ("env", "args"):
        print("\n".join(base.settings(MODE, False)[args.action == "args"]))
    elif args.action == "ready":
        base.ready(MODE, False, Path(values[2]))
    elif args.action == "arm":
        arm(Path(values[0]))
    elif args.action == "verify":
        observe(Path(values[2]))
    else:
        export(Path(values[2]), *(int(value) for value in values[3:]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Route diagnostic: {exc}") from None
