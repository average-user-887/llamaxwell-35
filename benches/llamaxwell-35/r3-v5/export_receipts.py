#!/usr/bin/env python3
"""Export release-safe numeric evidence from an offline campaign directory."""

import argparse
import hashlib
import json
from pathlib import Path
import re

PAIRS = (
    ("v5-top16-16k", "published-v5", "evening-midselector-bench", "16384-20261126T173206Z"),
    ("v5-top16-40k", "published-v5", "evening-midselector-bench", "40000-20261126T173703Z"),
    ("v5-held-out-16k", "published-v5", "evening-v5-heldout", "16512-20261126T175154Z"),
    ("v4-stack-2k", "earlier-v4", "evening-v4-shortbundle", "2000-20261126T170443Z"),
    ("v4-stack-40k", "earlier-v4", "evening-v4-longbundle", "40000-20261126T171134Z"),
    ("v4-fixed-stride-16k", "earlier-v4", "evening-widestride-bench", "16384-20261126T164902Z"),
    ("v4-fixed-stride-40k", "earlier-v4", "evening-widestride-bench", "40000-20261126T165413Z"),
    ("j48-16k-forward", "earlier-j48", "evening-mmq-j48-bench", "16384-20261126T161503Z"),
    ("j48-16k-reverse", "earlier-j48", "evening-j48-crossover", "16384-20261126T163100Z"),
    ("experimental-range-8k", "experimental-range", "evening-stride-range-bench", "8192-20261126T172903Z"),
    ("experimental-range-56k", "experimental-range", "evening-stride-range-long", "56000-20261126T174411Z"),
)
ARMS = ("off", "on")
RUNTIME_FILES = (
    "llama-server",
    "libllama.so.0.6.0",
    "libggml-cuda.so",
    "libllama-common.so.0.6.0",
    "libllama-server-impl.so",
)
REQUEST_KEYS = ("temperature", "seed", "cache_prompt", "ignore_eos", "stream")
EXACT_KEYS = ("n_out", "cache_n", "prompt_n", "draft_n", "draft_n_accepted")
TIMING_KEYS = ("ttft_s", "wall_s", "decode_s", "decode_toks_s", "prompt_ms")
EXTEND_STRIDE = "GGML_CUDA_OCCAMY_FA_Q2_R2_EXTEND_STRIDE"
RUNTIME_FEATURES = ("GGML_CUDA_MAXWELL_GRAPHS", "GGML_CUDA_OCCAMY_GDN_T2_Q8A")
SELECTOR_OPERATOR = "evening-top16-selector-bench/top16-selector-20261126T172518Z/comparison.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def digest(value):
    require(re.fullmatch("[0-9a-f]{64}", value), "Invalid SHA-256")
    return value


def numbers(values, keys):
    result = {key: values[key] for key in keys}
    require(all(type(value) in (int, float) for value in result.values()), "Non-numeric metric")
    return result


def env_value(value):
    require(str(value) in ("0", "1"), "Unexpected switch value")
    return int(value)


def binary_hashes(path):
    server = (path / "runtime-executable-hash.txt").read_text().split()[0]
    lines = (path / "runtime-library-hashes.txt").read_text().splitlines()
    hashes = dict(
        (Path(location).name, digest(value))
        for value, location in (line.split(maxsplit=1) for line in lines)
        if Path(location).name in RUNTIME_FILES
    )
    hashes["llama-server"] = digest(server)
    return {name: hashes[name] for name in RUNTIME_FILES}


def clock_summary(sample):
    if not sample or not sample["unique_fleet_samples"]:
        return None
    metrics = sample["metrics"]
    p_states = sample["pstates"]
    require(all(re.fullmatch(r"P[0-9]{1,2}", state) for state in p_states), "Invalid p-state")
    return {
        "samples": int(sample["unique_fleet_samples"]),
        "p_states": p_states,
        "median_graphics_mhz": metrics["graphics_clock_mhz"]["median"],
        "median_memory_mhz": metrics["memory_clock_mhz"]["median"],
        "median_temp_c": metrics["temp_c"]["median"],
    }


def dispatch_summary(receipt, arm):
    if "arms" in receipt:
        return numbers(receipt["arms"][arm], ("width", "pitch_bytes", "splits", "selector_markers", "stride_launches"))
    runtime = receipt.get("arm_runtime", receipt.get("live_bundle", {})).get(arm, {})
    summary = {}
    for name in ("first_stride_dispatch", "first_selector_dispatch", "first_dispatch"):
        dispatch = runtime.get(name)
        if dispatch:
            summary[name] = numbers(dispatch, sorted(set(dispatch) & {"width", "pitch_bytes", "kv_pitch", "splits", "plans"}))
    summary.update(numbers(runtime, sorted(set(runtime) & {"q2_stride_launches", "stride_host_dispatches", "midstride_dispatch"})))
    return summary or None


def read_pair(root, name, scope, directory, stamp, manifest, runtimes):
    path = root / directory / ("kv-tail-" + stamp)
    receipt, comparison_hash = load(path / "comparison.json")
    plan, plan_hash = load(path / "plan.json")
    require(receipt["exact_trajectory_parity"] is True, f"Trajectory mismatch for {name}")
    require(sorted(plan["arm_order"]) == list(ARMS), f"Unexpected arm order for {name}")

    runtime = binary_hashes(path / "off")
    require(runtime == binary_hashes(path / "on"), f"Runtime difference for {name}")
    require(runtime["llama-server"] == plan["server_sha256"] == manifest["released_server_sha256"], f"Server mismatch for {name}")
    require(runtime["libllama.so.0.6.0"] == receipt["host"]["sha256"] == manifest["runtime_overlays"]["libllama.so.0.6.0"]["sha256"], f"Host mismatch for {name}")
    require(runtime["libggml-cuda.so"] == receipt["cuda"]["sha256"], f"CUDA mismatch for {name}")
    if scope == "published-v5":
        require(runtime["libggml-cuda.so"] == manifest["runtime_overlays"]["libggml-cuda.so"]["sha256"], f"Release CUDA mismatch for {name}")
    require(runtimes.setdefault(scope, runtime) == runtime, f"Scope runtime mismatch for {name}")
    if "verified_sha256" in plan["model"]:
        require(plan["model"]["verified_sha256"] == manifest["model_sha256"], f"Model hash mismatch for {name}")

    allowed_flags = set(manifest["flags"]) | {EXTEND_STRIDE}
    require(set(plan["toggled_env"]) <= allowed_flags, f"Unexpected changed switch for {name}")
    changes = {
        flag: {arm: env_value(settings[arm]) for arm in ARMS}
        for flag, settings in plan["toggled_env"].items()
    }
    fixed = {
        flag: env_value(value)
        for flag, value in plan["shared_env"].items()
        if flag in allowed_flags | set(manifest["fixed_flags"]) | set(RUNTIME_FEATURES)
        and flag not in changes
    }

    runtime_data = next(
        (receipt[key] for key in ("arm_runtime", "cuda_engagement") if key in receipt),
        receipt["engagement"],
    )
    phases = {}
    counts_match = True
    for phase in plan["phase_output_limits"]:
        if "arms" in receipt:
            arms = {arm: receipt["arms"][arm]["phases"][phase] for arm in ARMS}
        else:
            comparison = receipt["phases"][phase]
            require(
                all(comparison[key] is True for key in ("identical_output_ids", "identical_text", "same_draft_proposed_and_accepted")),
                f"Phase comparison failed for {name}/{phase}",
            )
            arms = {arm: comparison[arm] for arm in ARMS}
        require(arms["off"]["output_sha256"] == arms["on"]["output_sha256"], f"Output hash mismatch for {name}/{phase}")
        counts = numbers(arms["off"], EXACT_KEYS)
        counts_match &= counts == numbers(arms["on"], EXACT_KEYS)
        timings = {arm: numbers(stats, TIMING_KEYS) for arm, stats in arms.items()}
        clocks = {}
        for arm in ARMS:
            arm_info = runtime_data.get(arm, {})
            source = (
                (arm_info.get("prefill_fleet") or arm_info.get("cold_fleet"))
                if phase == "cold"
                else receipt["engagement"][arm].get("decode_fleet", {}).get(phase)
            )
            clocks[arm] = clock_summary(source)
        phases[phase] = {
            "token_counts_both_arms": counts,
            "timings": timings,
            "clock_sample_summaries": clocks,
            "decode_change_pct": 100 * (timings["on"]["decode_toks_s"] / timings["off"]["decode_toks_s"] - 1),
        }
    require(counts_match, f"Token or cache counters differ for {name}")

    arguments = plan["server_args"]
    for flag, value in {
        "-np": "1",
        "-c": "84224",
        "-b": "2048",
        "-ub": "1024",
        "--spec-type": "draft-mtp",
        "--spec-draft-n-max": "1",
        "--spec-draft-type-k": "f16",
        "--spec-draft-type-v": "f16",
    }.items():
        require(arguments[arguments.index(flag) + 1] == value, f"Unexpected {flag} for {name}")

    reference = receipt.get("qualified_v4_reference", {}).get("comparison_sha256")
    if name == "v5-held-out-16k":
        reference = receipt["reference"]["hashes"]["comparison.json"]
    return {
        "id": name,
        "runtime": scope,
        "source_comparison_sha256": comparison_hash,
        "source_plan_sha256": plan_hash,
        "qualified_v4_comparison_sha256": digest(reference) if reference else None,
        "arm_order": plan["arm_order"],
        "prompt_tokens": int(plan["fixture"]["prompt_tokens"]),
        "request": {key: plan["request_template"][key] for key in REQUEST_KEYS},
        "context_cells": int(plan["context_cells"]),
        "changed_switches": changes,
        "fixed_switches": fixed,
        "exact_output_trajectory": True,
        "exact_counts": counts_match,
        "matched_output_tokens_per_arm": sum(phase["token_counts_both_arms"]["n_out"] for phase in phases.values()),
        "dispatch": {arm: dispatch_summary(receipt, arm) for arm in ARMS},
        "phases": phases,
    }


def selector_operator(root):
    cached, source_hash = load(root / SELECTOR_OPERATOR)
    require(cached["complete"] is True, "Selector operator matrix incomplete")
    shapes = []
    for fixture in cached["fixtures"]:
        shape = numbers(fixture, ("width", "pitch", "cases"))
        shape["direct_cases"] = len(fixture["direct_cases"])
        for name in ("selector_only", "complete_fa"):
            pair = fixture[name]
            shape[name] = {
                "generic_median_ms": pair["generic"]["median_ms"],
                "top16_median_ms": pair["top16"]["median_ms"],
                "top16_latency_change_pct": pair["top16_latency_change_pct"],
                "paired_top16_wins": pair["paired_top16_wins"],
            }
        shapes.append(shape)
    return {
        "source_comparison_sha256": source_hash,
        "exact_cases": sum(shape["cases"] + shape["direct_cases"] for shape in shapes),
        "shapes": shapes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="local directory with saved comparison and plan files")
    root = parser.parse_args().campaign
    manifest, manifest_hash = load(root / "rollup-v5/manifest.json")
    runtimes = {}
    pairs = [read_pair(root, *case, manifest, runtimes) for case in PAIRS]
    print(json.dumps({
        "schema_version": 1,
        "hardware": {"gpu": "NVIDIA Quadro M6000", "compute_capability": "5.2", "vram_gib": 24},
        "toolkit": "CUDA 12.8",
        "model": {"file": "occamy-1.0-Q4_0-headq40b-mtp.gguf", "sha256": digest(manifest["model_sha256"])},
        "base_source_commit": manifest["source_commit"],
        "source_manifest_sha256": manifest_hash,
        "source_patch_sha256_in_order": [digest(entry["sha256"]) for entry in manifest["source_patches_in_order"]],
        "runtimes": runtimes,
        "paired_model_runs": pairs,
        "standalone_top16_operator": selector_operator(root),
    }, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
