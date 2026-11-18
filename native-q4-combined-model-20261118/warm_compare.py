#!/usr/bin/env python3
"""Compare literal host-clock differences over equal-length warm intervals."""
import argparse
import json
from pathlib import Path

from trace_export import COST_NAMES, read_json


def time_us(clock):
    minute, second, ms, us = map(int, clock.split("."))
    return 60000000 * minute + 1000000 * second + 1000 * ms + us


def warm_anchors(path):
    meta = read_json(path / "trace-v2-meta.json")
    anchors, round_index, last_cycle = {}, 0, None
    complete, incomplete = 0, 0
    for line in (path / "trace-v2.jsonl").open():
        row = json.loads(line)
        if row.get("phase") != "warm":
            continue
        if row["event"] == "cycle" and "literal_restore_marker" in row:
            round_index += 1
            last_cycle = row
        elif row["event"] == "host_counter_snapshot":
            complete += 1
            if row["rounds"] % 25 == 0 and last_cycle and last_cycle["log_clock"]:
                anchors[round_index] = (row, last_cycle["log_clock"])
        elif row["event"] == "host_counter_record_incomplete":
            incomplete += 1
    return anchors, meta, {"native_rounds": round_index, "complete_snapshots": complete,
                           "incomplete_snapshot_records": incomplete}


def interval(begin, end, count):
    a, t0 = begin
    b, t1 = end
    dt = (time_us(t1) - time_us(t0)) / 1000
    if b["rounds"] - a["rounds"] != count or dt <= 0 or a["speculative_restores"] != b["speculative_restores"]:
        return None
    if sum(b["accepted_prefix_histogram"]) - sum(a["accepted_prefix_histogram"]) != count:
        return None
    scopes = {}
    for name in COST_NAMES:
        x, y = a["reported_costs"][f"{name}_ms"], b["reported_costs"][f"{name}_ms"]
        duration = y["cumulative_ms"] - x["cumulative_ms"]
        if y["calls"] - x["calls"] != count or duration < 0:
            return None
        scopes[name] = duration / count
    clock_ms = dt / count
    return {"process_round_anchors": [a["rounds"], b["rounds"]],
            "clock_ms_per_round": clock_ms, "host_scope_ms_per_round": scopes,
            "clock_residual_ms_per_round": clock_ms - sum(scopes.values()),
            "direct_rollback_ms_per_round": None, "checkpoint_restore_delta": 0}


def compare(off, on):
    checks = [warm_anchors(path) for path in (off, on)]
    left, right = (row[0] for row in checks)
    options = [(abs(a - b), -(end - a), a, b) for a in left for b in right if abs(a - b) <= 12
               for end in left if end > a and b + end - a in right]
    for _, negative_count, first_off, first_on in sorted(options):
        count = -negative_count
        first = (first_off, first_on)
        intervals = [interval(anchors[start], anchors[start + count], count)
                     for start, (anchors, _, _) in zip(first, checks)]
        if all(intervals):
            arms = [{"arm": meta["arm"], "technical_pass": meta["technical_pass"], "runner_exit": meta["runner_exit"],
                     "cleanup_exit": meta["cleanup_exit"], "warm_round_anchors": [start, start + count],
                     **seen, **measurement}
                    for start, (_, meta, seen), measurement in zip(first, checks, intervals)]
            difference = {key: arms[0]["host_scope_ms_per_round"][key] - arms[1]["host_scope_ms_per_round"][key]
                          for key in COST_NAMES}
            return {"scope": "complete process-cumulative counters over equal-length warm intervals",
                    "warm_anchor_start_offset_rounds": first_on - first_off, "round_span": count, "arms": arms,
                    "off_minus_on_host_scope_ms_per_round": difference,
                    "off_minus_on_accept_clock_ms_per_round": arms[0]["clock_ms_per_round"] - arms[1]["clock_ms_per_round"]}
    raise ValueError("no near-aligned equal-length warm interval with complete scope calls and round accounting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.off, args.on), indent=2, allow_nan=False))
