"""Recompute smoke summaries and compact counter observations from public rows."""

from collections import Counter
from json import loads
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "workers"))

from compact_witness import marker
from ifeval_pairs import add_counts, empty_tally, with_rates

GUARDS = ("ready", "observer", "types", "rollback", "shape", "model", "draft", "sampler", "mask")
COMPACT = ("requested", "launched", "compact_only", "ambiguous", "unsupported", "raw_bytes",
           "compact_bytes", "shadow_rows", "shadow_mismatches")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return loads((ROOT / "evidence" / name).read_text())


def verify_bfcl_scores(score):
    results = {}
    for mode, arm in score["arms"].items():
        cases = arm["cases"]
        results[mode] = {case["id"]: case["passed"] for case in cases}
        require(len(cases) == len(results[mode]) == 4, "BFCL public case coverage")
        require(sum(results[mode].values()) == arm["pass_count"], "BFCL passes")
        require(all(not case["infra_error"] and case["limit"] is None for case in cases),
                "BFCL infrastructure or cap event")
        require(all(not case["tool_message_error_kinds"] and not case["tool_event_error_kinds"]
                    for case in cases), "BFCL typed tool error")
    native, serial = results["native"], results["serial"]
    require(native.keys() == serial.keys(), "BFCL paired IDs")
    diff = Counter(native[case] - serial[case] for case in native)
    require(score["paired"] == {"native_wins": diff[1], "native_losses": diff[-1], "ties": diff[0]},
            "BFCL paired outcomes")


def verify_bfcl_counters(score):
    data = read("bfcl-counter-rows.json")
    for mode, rows in data["arms"].items():
        witness = score["arms"][mode]["physical"]
        for label, keys in (("compact_guard", GUARDS), ("compact_readback", COMPACT)):
            pattern = marker(label, keys)
            observed = {}
            for step, lines in rows.items():
                selected = [line for line in lines if f"{label}:" in line]
                matches = [pattern.fullmatch(line) for line in selected]
                require(all(matches), f"Malformed {label}")
                history = [tuple(map(int, row.groups())) for row in matches]
                require(all(all(a <= b for a, b in zip(old, new, strict=True))
                            for old, new in zip(history, history[1:])), f"Non-monotonic {label}")
                observed[step] = history[-1] if history else (0,) * len(keys)
                if mode == "native" and step == "terminal":
                    forms = Counter("bare" if line.startswith(label) else
                                    "timestamp_info" if " I " in line else "timestamp_only"
                                    for line in selected)
                    inherited = forms["bare"] + forms["timestamp_info"]
                    require(score["compact_formats"][label] == {
                        **forms, "total": len(selected), "inherited": inherited,
                        "corrected": len(history),
                    }, f"BFCL original and corrected {label} forms")
            delta = {key: end - start for key, end, start in zip(
                keys, observed["terminal"], observed["startup"], strict=True)}
            require(delta == witness["compact_guard" if label == "compact_guard" else "compact"],
                    f"BFCL {label} counters")


def verify_ifeval(score):
    native = score["arms"]["native"]["cases"]
    serial = score["arms"]["serial"]["cases"]
    require(native.keys() == serial.keys() == {str(i) for i in score["public_ids"]}, "IFEval paired IDs")
    for mode in ("strict", "loose"):
        prompts, instructions = empty_tally(), empty_tally()
        for case in native:
            off, on = serial[case], native[case]
            require(off["status"] == on["status"] == "scored", "IFEval incomplete case")
            add_counts(prompts, all(off[mode]), all(on[mode]))
            for a, b in zip(off[mode], on[mode], strict=True):
                add_counts(instructions, a, b)
        require(with_rates(prompts) == score["metrics"][mode]["prompt"], "IFEval prompt summary")
        require(with_rates(instructions) == score["metrics"][mode]["instruction"],
                "IFEval instruction summary")


def verify_fallback(score):
    for arm in score["arms"].values():
        observation = arm["physical"]
        compact = observation["compact"]
        require(all(compact[key] == 0 for key in ("requested", "launched", "compact_only")),
                "Compact launches appeared")
        offers = sum(observation["compact_guard"].values())
        require(observation["observed_fallback_proposals"] == offers
                and observation["down4_launches"] == 40 * offers, "Fallback counter mismatch")


def main():
    bfcl = read("bfcl-uncached-score.json")
    ifeval = read("ifeval-smoke-score.json")
    verify_bfcl_scores(bfcl)
    verify_bfcl_counters(bfcl)
    verify_ifeval(ifeval)
    verify_fallback(bfcl)
    verify_fallback(ifeval)
    print("Verified paired BFCL and IFEval outcomes and fallback counters.")


if __name__ == "__main__":
    main()
