#!/usr/bin/env python3
"""Export replayable typed trace events from a completed private receipt, including failures."""
import argparse
import json
import re
from pathlib import Path

import native_probe as n

COST = re.compile(r"\b(draft_ms|tgt_decode_ms|spec_process_ms|sample_accept_ms)=([\d.]+)\(n=(\d+)\)")
STAMP = re.compile(r"(\d+\.\d{2}\.\d{3}\.\d{3})\s+[A-Z]\s+slot\b")
STOP = re.compile(r"\bstop processing: n_tokens = (\d+), truncated = ([01])")
PHYSICAL = re.compile(r"restored context checkpoint .*?n_past = (\d+)|restoring speculative checkpoint|context shift, n_keep")


def read_phase_record(raw, phase):
    if phase not in n.STAGES:
        return {}
    return n.partial_record(raw / f"{phase}.json")


def annotated(mode, raw):
    yield {"event": "trace_metadata", "arm": mode, "format": 2,
           "scope": "literal typed events plus explicitly labeled emission reconciliation; no token IDs or text"}
    try:
        start, cold, warm, full = n.log_phases(raw)
        logs = {"startup":start,"cold":cold,"warm":warm}
        if mode.startswith("smoke") and (raw/"server-post-warm.log").is_file():
            point=(raw/"server-post-warm.log").read_text(errors="replace")
            before=(raw/"server-pre-warm.log").read_text(errors="replace")
            if point.startswith(before) and full.startswith(point):
                logs["warm"]=point[len(before):];logs["quality"]=full[len(point):]
        if not full:
            yield {"event": "logs_unavailable", "arm": mode}
    except ValueError:
        path = raw / "server.log"
        logs = {"unattributed": path.read_text(errors="replace") if path.is_file() else ""}
        yield {"event": "log_discontinuity", "arm": mode, "request_attribution": None}
    count = 0
    for phase, log in logs.items():
        rec = read_phase_record(raw, phase)
        if mode.startswith("smoke") and phase in ("cold","warm") and isinstance(rec.get("token_ids"),list):
            rec["n_out"]=len(rec["token_ids"]);rec["n_predict"]=192
        total = rec.get("n_out") if type(rec.get("n_out")) is int and 0 <= rec["n_out"] <= 3072 else None
        requested = rec.get("n_predict") if rec.get("n_predict") in (192, 3072) else None
        finished = rec.get("error") is None and rec.get("finish_reason") == "limit" and total == requested
        if phase in n.STAGES:
            yield {"event": "request_begin", "arm": mode, "phase": phase, "requested": requested,
                   "record_available": bool(rec), "output_count": total, "finished_at_limit": finished}
        all_rows, residual = n.cycles(mode, phase, log, total or 0, finished)
        serial = not n.MODES[mode]
        if serial:
            for row in all_rows:
                count += 1
                yield {"event": "cycle", "arm_cycle": count, **row}
        elif finished and residual:
            # Prompt sampling emits the first serial token. Any second residual remains unattributed.
            count += 1
            yield {"event": "cycle", "arm_cycle": count, "cycle": 0, "arm": mode, "phase": phase,
                   "offered": 0, "accepted": 0, "emitted": 1, "rejected": 0,
                   "state_action": "first_token_from_prompt", "emission_provenance": "native_source_and_total_reconciliation",
                   "target_ms": None, "draft_ms": None, "accept_ms": None, "cycle_ms": None}
        native_rows = iter(all_rows if not serial else [])
        for lineno, line in enumerate(log.splitlines(), 1):
            location = {"arm": mode, "phase": phase, "relative_log_line": lineno}
            for match in n.original.n92.ACCEPT.finditer(line):
                if serial:
                    yield {"event": "unexpected_native_accept", **location, "accepted": int(match[1]),
                           "offered": int(match[2]), "literal_restore_marker": bool(match[3])}
                    continue
                row = next(native_rows)
                count += 1
                stamp = STAMP.search(line[max(0, match.start() - 120):match.start()])
                yield {"event": "cycle", "arm_cycle": count, "log_clock": stamp[1] if stamp else None,
                       "literal_restore_marker": bool(match[3]), **location, **row}
            if "mv1_host: rounds=" in line:
                counter = n.HOST.fullmatch(line)
                if not counter:
                    yield {"event": "host_counter_record_incomplete", **location}
                    continue
                values = {name: {"cumulative_ms": float(value), "calls": int(calls)}
                          for name, value, calls in COST.findall(line)}
                yield {"event": "host_counter_snapshot", **location, "rounds": int(counter[1]),
                       "speculative_restores": int(counter[2]), "reported_costs": values,
                       "cost_scope": "process cumulative; individual cycle cost unknown"}
            for physical in PHYSICAL.finditer(line):
                kind = ("prompt_cache_checkpoint" if physical[1] else
                        "speculative_checkpoint_restore" if "speculative" in physical[0] else "context_shift")
                yield {"event": "runtime_state_marker", **location, "kind": kind,
                       "restored_prompt_tokens": int(physical[1]) if physical[1] else None}
            for stop in STOP.finditer(line):
                yield {"event": "slot_release", **location, "resident_slot_tokens": int(stop[1]),
                       "context_truncated": bool(int(stop[2]))}
        if phase in n.STAGES:
            points = rec.get("observed_sse_checkpoints")
            if isinstance(points, list):
                for point in points:
                    keys = ("output_token", "elapsed_s", "event_first_token", "event_last_token")
                    if isinstance(point, dict) and all(type(point.get(key)) in (int, float) for key in keys):
                        yield {"event": "sse_clock_checkpoint", "arm": mode, "phase": phase,
                               **{key: point[key] for key in keys}}
            reason = rec.get("finish_reason")
            yield {"event": "output_boundary", "arm": mode, "phase": phase, "observed_output_count": total,
                   "requested_output_limit": requested, "limit_stop_observed": finished,
                   "context_truncated": rec.get("truncated") if type(rec.get("truncated")) is bool else None,
                   "serial_emitted_unassigned_to_cycle": max(residual - 1, 0)
                   if residual is not None and not serial else residual,
                   "finish_reason": reason if reason in ("limit", "eos", "word") else None,
                   "request_wall_s": rec.get("wall_s") if type(rec.get("wall_s")) in (int, float) else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    raw = receipt / "private"
    status = n.n9.key_values(raw / "status.txt")
    manifest = n.n9.key_values(raw / "manifest.txt")
    if "completed_epoch" not in status or manifest["mode"] not in n.MODES:
        raise ValueError("supply a completed typed receipt")
    names = ("private/trace-v2.jsonl", "sanitized/trace-v2.jsonl")
    if any((receipt / name).exists() for name in names):
        raise ValueError("trace-v2 already exists")
    with (receipt / names[0]).open("x") as private, (receipt / names[1]).open("x") as public:
        for row in annotated(manifest["mode"], raw):
            line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            private.write(line)
            public.write(line)
    n.n9.write_json(receipt / "sanitized/trace-v2-meta.json", {
        "format": 2, "trace_sha256": n.n9.hash_file(receipt / names[1]),
        "private_trace_sha256": n.n9.hash_file(receipt / names[0]),
        "runner_exit": int(status["runner_exit"]), "cleanup_exit": int(status["cleanup_exit"]),
        "converter_sha256": n.n9.hash_file(Path(__file__)),
    })


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise SystemExit(f"Trace export: {exc}") from None
