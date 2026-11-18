#!/usr/bin/env python3
"""Export the native-width trace-v2 chronology from completed cards, read-only."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEPTHS = {"no-mtp": 0, "no-mtp-on": 0, "serial-on": 0, "serial-off": 0, "depth1": 1, "depth2": 2,
          "depth2-off": 2, "depth2-on": 2, "depth3": 3, "depth3-off": 3, "depth3-on": 3,
          "depth3-attn": 3, "depth3-combined": 3}
MODES = {
    "resume-native-width": {"no-mtp", "depth1", "depth2", "depth3"},
    "resume-native-cycle-timing": {"depth1", "depth2", "depth3"},
    "resume-q3-moe-native": {"no-mtp", "depth2-off", "depth2-on", "smoke-off", "smoke-on"},
    "resume-q3-moe-native-r1": {"no-mtp", "depth2-off", "depth2-on", "smoke-off", "smoke-on"},
    "resume-q3-moe-native-r2": {"no-mtp", "no-mtp-on", "depth2-off", "depth2-on", "smoke-off", "smoke-on"},
    "resume-q4-r2-native-r2": {"no-mtp", "no-mtp-on", "depth3-off", "depth3-on"},
    "resume-q4-combined-native": {"serial-on", "depth3-off", "depth3-attn", "depth3-combined", "smoke-off", "smoke-on"},
}
ACCEPT = re.compile(r"\baccepted\s+(\d+)/\s*(\d+) draft tokens\b( \(restore checkpoint\))?")
STAMP = re.compile(r"(\d+\.\d{2}\.\d{3}\.\d{3})\s+[A-Z]\s+slot\b")
STOP = re.compile(r"\bstop processing: n_tokens = (\d+), truncated = ([01])")
PHYSICAL = re.compile(r"restored context checkpoint .*?n_past = (\d+)|restoring speculative checkpoint|context shift, n_keep")
COST_NAMES = ("draft", "tgt_decode", "spec_process", "sample_accept")
COST = re.compile(r"\b(draft|tgt_decode|spec_process|sample_accept)_ms=([\d.]+)\(n=(\d+)\)")
HOST = re.compile(r"mv1_host: rounds=(\d+) restores=(\d+)" +
                  "".join(rf" {name}_ms=[\d.]+\(n=\d+\)" for name in COST_NAMES) +
                  r" acc_hist=\[(\d+),(\d+),(\d+),(\d+)\]")
FINISH_REASONS = ("limit", "eos", "word")
SELECTOR_FIRST = re.compile(r"^q2_r2_top16_dispatch n_q=2 n_kv=(\d+) nkvh=(\d+) plans=(\d+)$")
SPARSE = re.compile(r"^ggml_cuda: occamy_sparse_probe dispatch calls=(\d+) n_q=(\d+) n_kv=(\d+) nkvh=(\d+) "
                    r"nsplit=(\d+) blocks_per_sm=(\d+) approximate=1 keep_old_blocks=(\d+) recent=(\d+) revision=(\d+)$")
OFFLOAD = re.compile(r"\bload_tensors: offloaded (\d+)/(\d+) layers to GPU$")



def read_text(path):
    return path.read_text(errors="replace") if path.is_file() else ""


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def fields(path):
    rows = [line.split("=", 1) for line in path.read_text().splitlines()]
    if any(len(row) != 2 for row in rows) or len(dict(rows)) != len(rows):
        raise ValueError("invalid typed status or manifest")
    return dict(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_info(receipt):
    if receipt.parent.name != "receipts":
        raise ValueError("use an original completed typed receipt")
    card = receipt.parent.parent.name
    raw = receipt / "private"
    status = fields(raw / "status.txt")
    mode = fields(raw / "manifest.txt")["mode"]
    if mode not in MODES.get(card, ()) or "completed_epoch" not in status:
        raise ValueError("unsupported card, mode or unfinished receipt")
    runner, cleanup = int(status["runner_exit"]), int(status["cleanup_exit"])
    reported = read_json(receipt / "sanitized/receipt.json").get("status", {})
    if reported and (reported["runner_exit"], reported["cleanup_exit"]) != (runner, cleanup):
        raise ValueError("original status records differ")
    reason = None
    if "observer did not see existing Q1/Q2 dispatches" in read_text(raw / "run.log"):
        reason = "observer_q1_q2_attribution_gate"
    return mode, raw, 3 if card == "resume-q4-combined-native" and mode.startswith("smoke") else DEPTHS.get(mode, 2), {
        "runner_exit": runner, "cleanup_exit": cleanup, "technical_pass": reported.get("technical_pass"),
        "failure_class": reason, "bounded_wall_s": reported.get("bounded_wall_s"),
    }


def phases(raw, smoke):
    log_names = ("server-start.log", "server-pre-warm.log", "server-post-warm.log", "server.log")
    start, prewarm, postwarm, end = (read_text(raw / name) for name in log_names)
    fallback = end or postwarm or prewarm or start
    if not fallback:
        return {}, "logs_unavailable"
    if not start or any(not longer.startswith(shorter) for shorter, longer in
                        zip([v for v in (start, prewarm, postwarm, end) if v],
                            [v for v in (start, prewarm, postwarm, end) if v][1:])):
        return {"unattributed": fallback}, "log_discontinuity"
    if smoke and read_json(raw / "quality-response.json") and not postwarm:
        return {"unattributed": fallback}, "quality_log_boundary_missing"
    logs = {"startup": start}
    if prewarm:
        logs["cold"] = prewarm[len(start):]
        if postwarm:
            logs["warm"] = postwarm[len(prewarm):]
            if end:
                logs["quality"] = end[len(postwarm):]
        elif end:
            logs["warm"] = end[len(prewarm):]
    elif end:
        logs["cold"] = end[len(start):]
    # The terminal cumulative dump is process shutdown evidence and belongs outside request timing.
    last_phase = next(reversed(logs))
    last_log = logs[last_phase]
    stops = list(STOP.finditer(last_log)) if last_phase != "startup" else []
    if stops:
        line_end = last_log.find("\n", stops[-1].end())
        if line_end >= 0:
            logs[last_phase] = last_log[:line_end + 1]
            logs["shutdown"] = last_log[line_end + 1:]
    return logs, None if end else "final_log_unavailable"


def finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def request(raw, phase, smoke):
    record = read_json(raw / ("quality-response.json" if phase == "quality" else f"{phase}.json"))
    timing = record.get("timings") or {}
    quality = phase == "quality"
    if quality:
        requested = read_json(raw / "quality-request.json").get("n_predict")
        total = record.get("tokens_predicted")
        reason = record.get("stop_type")
    elif smoke:
        requested = 192
        total = len(record["token_ids"]) if isinstance(record.get("token_ids"), list) else None
        reason = record.get("finish_reason")
    else:
        requested, total, reason = (record.get(k) for k in ("n_predict", "n_out", "finish_reason"))
    valid_total = total if type(total) is int and 0 <= total <= 3072 else None
    requested = requested if type(requested) is int and 1 <= requested <= 3072 else None
    completed = bool(record and record.get("error") is None and reason in FINISH_REASONS and valid_total is not None and
                     valid_total == timing.get("predicted_n") and record.get("truncated") is False and
                     requested is not None and 0 <= valid_total <= requested and (reason != "limit" or valid_total == requested))
    return record, valid_total, requested, reason if reason in FINISH_REASONS else None, completed


def clock_of(line, start):
    stamps = list(STAMP.finditer(line[:start]))
    return stamps[-1][1] if stamps else None


def host_snapshot(line):
    match = HOST.fullmatch(line)
    if not match:
        return None
    return {"rounds": int(match[1]), "speculative_restores": int(match[2]),
            "accepted_prefix_histogram": [int(value) for value in match.groups()[2:]],
            "reported_costs": {f"{name}_ms": {"cumulative_ms": float(ms), "calls": int(calls)}
                               for name, ms, calls in COST.findall(line)},
            "cost_scope": "process cumulative; individual cycle cost unknown"}


def annotated(mode, raw, depth):
    yield {"event": "trace_metadata", "arm": mode, "format": 2,
           "scope": "literal typed events plus labeled stream-emission reconciliation"}
    logs, issue = phases(raw, mode.startswith("smoke"))
    if issue:
        yield {"event": issue, "arm": mode}
    count = 0
    for phase, log in logs.items():
        lane = {"arm": mode, "phase": phase}
        has_request = phase in ("cold", "warm", "quality")
        record, total, requested, reason, finished = request(raw, phase, mode.startswith("smoke")) if has_request else (
            {}, None, None, None, False)
        accepted = list(ACCEPT.finditer(log))
        residual = total - sum(int(row[1]) + 1 for row in accepted) if total is not None else None
        reconcile = bool(depth and accepted and finished and residual in (1, 2) and not any(row[3] for row in accepted))
        if has_request:
            yield {"event": "request_begin", **lane, "requested": requested, "record_available": bool(record),
                   "output_count": total, "finished_at_limit": finished and reason == "limit"}
        if not depth and total:
            for serial_index in range(1, total + 1):
                count += 1
                yield {"event": "cycle", **lane, "arm_cycle": count, "cycle": serial_index, "offered": 0,
                       "accepted": 0, "rejected": 0, "emitted": 1, "state_action": "serial_output",
                       "emission_provenance": "observed_stream_ID", "cycle_ms": None, "replay": None}
        if reconcile:
            count += 1
            yield {"event": "cycle", **lane, "arm_cycle": count, "cycle": 0, "offered": 0, "accepted": 0,
                   "rejected": 0, "emitted": 1, "state_action": "first_token_from_prompt",
                   "emission_provenance": "native_source_and_total_reconciliation", "cycle_ms": None, "replay": None}
        cycle = 0
        for lineno, line in enumerate(log.splitlines(), 1):
            loc = {**lane, "relative_log_line": lineno}
            if m := SELECTOR_FIRST.fullmatch(line):
                yield {"event": "fast_selector_first_dispatch", **loc, "key_width": int(m[1]),
                       "kv_heads": int(m[2]), "plans": int(m[3]), "query_rows": 2}
            if m := SPARSE.fullmatch(line):
                fields = ("calls", "query_rows", "key_width", "kv_heads", "splits", "blocks_per_sm",
                          "retained_old_blocks", "recent_keys", "revision")
                yield {"event": "sparse_attention_dispatch_sample", **loc, **dict(zip(fields, map(int, m.groups())))}
            if m := OFFLOAD.search(line):
                yield {"event": "offload_observed", **loc, "offloaded_layers": int(m[1]), "configured_layers": int(m[2])}
            for match in ACCEPT.finditer(line):
                kept, offered, restore = int(match[1]), int(match[2]), bool(match[3])
                if not depth:
                    yield {"event": "unexpected_native_accept", **loc, "accepted": kept, "offered": offered,
                           "literal_restore_marker": restore}
                    continue
                count += 1
                cycle += 1
                state = "checkpoint_restore_marker" if restore else "native_trim_path" if kept < offered else "all_accepted"
                yield {"event": "cycle", **loc, "arm_cycle": count, "cycle": cycle,
                       "log_clock": clock_of(line, match.start()), "accepted": kept, "offered": offered,
                       "rejected": offered - kept, "literal_restore_marker": restore, "emitted": kept + 1 if reconcile else None,
                       "emission_provenance": "native_accept_plus_one_reconciled_with_response_total" if reconcile else "unknown",
                       "state_action": state, "replay": None, "target_ms": None, "draft_ms": None, "accept_ms": None,
                       "cycle_ms": None, "cost_provenance": "no_per_cycle_clock"}
            if "mv1_host: rounds=" in line:
                host = host_snapshot(line)
                if host is None:
                    yield {"event": "host_counter_record_incomplete", **loc}
                else:
                    yield {"event": "host_counter_snapshot", **loc, **host}
            for physical in PHYSICAL.finditer(line):
                kind = ("prompt_cache_checkpoint" if physical[1] else
                        "speculative_checkpoint_restore" if "speculative" in physical[0] else "context_shift")
                yield {"event": "runtime_state_marker", **loc, "kind": kind,
                       "restored_prompt_tokens": int(physical[1]) if physical[1] else None}
            for stop in STOP.finditer(line):
                yield {"event": "slot_release", **loc, "resident_slot_tokens": int(stop[1]),
                       "context_truncated": bool(int(stop[2]))}
        if not has_request:
            continue
        points = record.get("observed_sse_checkpoints")
        if isinstance(points, list):
            keys = ("output_token", "elapsed_s", "event_first_token", "event_last_token")
            for point in points:
                if isinstance(point, dict) and all(finite(point.get(k)) is not None for k in keys):
                    yield {"event": "sse_clock_checkpoint", **lane, **{k: point[k] for k in keys}}
        if phase == "quality":
            score = read_json(raw / "quality-score.json")
            yield {"event": "arithmetic_score", **lane, **{k: score[k] for k in
                   ("schema_passed", "math_passed", "passed") if type(score.get(k)) is bool}}
        yield {"event": "output_boundary", **lane, "observed_output_count": total, "requested_output_limit": requested,
               "limit_stop_observed": finished and reason == "limit", "finish_observed": finished,
               "context_truncated": record.get("truncated") if type(record.get("truncated")) is bool else None,
               "serial_emitted_unassigned_to_cycle": residual - 1 if reconcile else 0 if not depth and total else None,
               "finish_reason": reason, "request_wall_s": finite(record.get("wall_s"))}


def export(receipt, label):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
        raise ValueError("use a portable lowercase label")
    mode, raw, depth, status = run_info(receipt.resolve())
    out = HERE / "exports" / label
    out.mkdir(parents=True, mode=0o700)
    trace = out / "trace-v2.jsonl"
    counts = Counter()
    with trace.open("x") as file:
        for row in annotated(mode, raw, depth):
            counts[row["event"]] += 1
            file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    meta = {"format": 2, "arm": mode, "depth": depth, **status, "trace_sha256": sha(trace),
            "converter_sha256": sha(Path(__file__)), "events": dict(sorted(counts.items()))}
    (out / "trace-v2-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    print(json.dumps(export(args.receipt, args.label), indent=2))


if __name__ == "__main__":
    main()
