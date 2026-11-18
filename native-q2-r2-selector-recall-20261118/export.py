"""Publish typed recall statuses and a count-only timeline from the owner-held receipt."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CARD = HERE.parents[1] / "last-ms-r2-selector-recall"
MODEL_PACKET = HERE.parent / "native-q2-r2-selector-model-20261118"

def load_mod(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module

trace = load_mod("safe_trace", MODEL_PACKET / "trace_export.py")
recall = load_mod("recall", CARD / "model.py")

def typed_log(text, phase):
    events = []
    for lineno, line in enumerate(text.splitlines(), 1):
        scope = {"phase": phase, "relative_log_line": lineno}
        if m := trace.OFFLOAD.search(line):
            events.append({"event": "offload", **scope, "layers": int(m[1]), "total_layers": int(m[2])})
        if m := trace.SELECTOR_FIRST.fullmatch(line):
            events.append({"event": "fast_selector_first_dispatch", **scope, "key_width": int(m[1]),
                           "kv_heads": int(m[2]), "plans": int(m[3]), "query_rows": 2})
        if m := trace.SPARSE.fullmatch(line):
            keys = ("calls", "query_rows", "key_width", "kv_heads", "splits", "blocks_per_sm",
                    "retained_old_blocks", "recent_keys", "revision")
            events.append({"event": "sparse_attention_dispatch_sample", **scope, **dict(zip(keys, map(int, m.groups())))})
        if "mv1_host: rounds=" in line:
            host = trace.host_snapshot(line)
            if host is not None:
                events.append({"event": "host_counter_snapshot", **scope, **host})
            else:
                events.append({"event": "host_counter_record_incomplete", **scope})
        if m := recall.n88.COUNTER.fullmatch(line):
            events.append({"event": "t2_counter_sample", **scope, "calls": int(m[1])})
        if line == recall.n88.ADMIT:
            events.append({"event": "process_q8_admission", **scope, "original_blocks": 3,
                           "candidate_blocks": 3, "grid": 72})
        for m in trace.ACCEPT.finditer(line):
            events.append({"event": "native_verifier_accept", **scope, "accepted": int(m[1]), "offered": int(m[2]),
                           "literal_restore_marker": bool(m[3])})
        for m in trace.PHYSICAL.finditer(line):
            kind = ("prompt_cache_checkpoint" if m[1] else
                    "speculative_checkpoint_restore" if "speculative" in m[0] else "context_shift")
            events.append({"event": "runtime_state_marker", **scope, "kind": kind,
                           "restored_prompt_tokens": int(m[1]) if m[1] else None})
        for m in trace.STOP.finditer(line):
            events.append({"event": "slot_release", **scope, "resident_slot_tokens": int(m[1]),
                           "context_truncated": bool(int(m[2]))})
    return events

def build(out):
    subprocess.run(["sha256sum", "--check", "--quiet", "observations.sha256"], cwd=out, check=True)
    summary = trace.read_json(out / "sanitized/model-receipt.json")
    status = trace.fields(out / "status.txt")
    assert all(int(status[k]) == summary[k] for k in ("runner_exit", "cleanup_exit"))
    success = summary["runner_exit"] == summary["cleanup_exit"] == 0
    assert not success or recall.verify(out) == summary["validation"]
    start = trace.read_text(out / "server-start.log")
    final = trace.read_text(out / "server.log")
    continuous = bool(start and final.startswith(start))
    request = trace.read_json(out / "request.json")
    scoring = trace.read_json(out / "score.json")
    events = [{"event": "trace_metadata", "format": 2, "arm": "native_D1_midpoint_recall",
               "transport": "one non-streaming response", "log_boundary_available": continuous}]
    events.extend(typed_log(start if continuous else "", "startup"))
    names = ("row0", "row1", "combined", "skip", "q2_sparse")
    def counter(event, filename):
        values = trace.read_json(out / "witness" / filename)
        if all(type(values.get(k)) is int for k in names):
            events.append({"event": "physical_counter_boundary", "boundary": event,
                           "counters": {key: values[key] for key in names}})
    counter("before_request", "start.json")
    if request or continuous:
        events.append({"event": "request_boundary_begin", "phase": "recall", "expected_prompt_tokens": 80000,
                       "reference_midpoint_token": 40000, "response_record_available": bool(request)})
    raw_phase = final[len(start):] if continuous else final
    shutdown = ""
    if continuous and (stops := list(trace.STOP.finditer(raw_phase))):
        line_end = raw_phase.find("\n", stops[-1].end())
        if line_end >= 0:
            raw_phase, shutdown = raw_phase[:line_end + 1], raw_phase[line_end + 1:]
    events.extend(typed_log(raw_phase, "recall" if continuous else "unattributed"))
    if request:
        timings = request["timings"]
        values = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted")
        safe_score = ("allowlist_match", "eos_terminated", "recall_pass", "n_out_including_eos")
        events.append({"event": "request_boundary_result", "phase": "recall", "wall_s": request["wall_s"],
                       "finish_reason": request["finish_reason"], "max_output": request["max_output"],
                       "server_timing": {k: timings.get(k) for k in values}, "score": {k: scoring.get(k) for k in safe_score}})
    counter("after_request", "cold.json")
    events.extend(typed_log(shutdown, "shutdown"))
    events.append({"event": "owned_run_outcome", "phase": "shutdown", "runner_exit": summary["runner_exit"],
                   "cleanup_exit": summary["cleanup_exit"], "bounded_wall_s": summary["bounded_wall_s"]})
    counts = Counter(row["event"] for row in events)
    assert counts["host_counter_snapshot"] + counts["host_counter_record_incomplete"] == final.count("mv1_host: rounds=")
    assert not success or (continuous and counts["physical_counter_boundary"] == 2 and counts["fast_selector_first_dispatch"] == 1)
    assert not success or counts["sparse_attention_dispatch_sample"] == final.count("ggml_cuda: occamy_sparse_probe dispatch calls=")
    encoded = "".join(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n" for event in events).encode()
    (HERE / "recall.trace-v2.jsonl").write_bytes(encoded)
    safe_error = summary["refusal"]
    if safe_error and "/" in safe_error:
        safe_error = "original technical refusal retained privately"
    summary["refusal"] = safe_error
    summary.update({"trace_sha256": sha256(encoded).hexdigest(), "typed_event_counts": dict(sorted(counts.items()))})
    (HERE / "recall.receipt.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        recall.check()
        archived = trace.read_text(recall.ARCHIVE / "server.log")
        typed = typed_log(archived, "archive")
        assert sum(r["event"] == "offload" for r in typed) == 1
        assert sum(r["event"] == "host_counter_snapshot" for r in typed) == archived.count("mv1_host: rounds=")
        assert any(r["event"] == "sparse_attention_dispatch_sample" and r["query_rows"] == 2
                   and r["key_width"] == 80128 for r in typed)
        print("Pinned archive score and typed count-only chronology revalidated offline.")
    elif len(sys.argv) == 2:
        build(Path(sys.argv[1]).resolve())
    else:
        raise SystemExit("usage: export.py --check | RECALL_RECEIPT")
