"""Export complete typed chronology for the selector's two original model arms."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import export
from trace_export import annotated, read_text

KEYS = {"row0", "row1", "combined", "skip", "q2_sparse"}
ARMS = ("serial", "native")

def render(mode, raw):
    export.verify_observations(raw)
    receipt = export.receipt(raw)
    assert receipt["runner_exit"] == receipt["cleanup_exit"] == 0
    assert receipt["mode"] == mode and receipt["validated"]["technical_pass"]
    snapshots = receipt["read_only_counter_snapshots"]
    assert set(snapshots) == {"start", "cold", "warm"}
    for words in snapshots.values():
        assert set(words) == KEYS and all(type(v) is int and v >= 0 for v in words.values())
    def boundary(label):
        return {"event": "physical_counter_boundary", "arm": mode, "boundary": label, "counters": snapshots[label]}
    events, status = [], Counter()
    for row in annotated(mode, raw, int(mode == "native")):
        if row["event"] == "request_begin" and row["phase"] == "cold":
            events.append(boundary("start"))
        events.append(row)
        if row["event"] == "output_boundary":
            phase = row["phase"]
            metrics = receipt["request"][phase]
            keys = ("ttft_s", "server_prompt_ms", "server_decode_ms", "offered_drafts", "accepted_drafts")
            events.append({"event": "observed_response_metrics", "arm": mode, "phase": phase,
                           **{key: metrics[key] for key in keys}})
            events.append(boundary(phase))
    events.append({"event": "owned_run_outcome", "arm": mode, "runner_exit": receipt["runner_exit"],
                   "cleanup_exit": receipt["cleanup_exit"], "duration_s": receipt["bounded_wall_s"]})
    status.update(row["event"] for row in events)
    raw_log = read_text(raw / "server.log")
    assert status["host_counter_snapshot"] == raw_log.count("mv1_host: rounds=")
    assert status["host_counter_record_incomplete"] == 0 and status["offload_observed"] == 1
    assert status["fast_selector_first_dispatch"] == int(mode == "native")
    assert status["sparse_attention_dispatch_sample"] == raw_log.count("ggml_cuda: occamy_sparse_probe dispatch calls=")
    for phase in ("cold", "warm"):
        response = export.read(raw / f"{phase}.json")
        ticks = response.get("observed_sse_checkpoints", [])
        assert sum(row["event"] == "sse_clock_checkpoint" and row["phase"] == phase for row in events) == len(ticks)
    assert status["physical_counter_boundary"] == 3 and status["output_boundary"] == 2
    assert status["log_discontinuity"] == status["final_log_unavailable"] == 0
    encoded = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in events).encode()
    name = f"{mode}.trace-v2.jsonl"
    (export.HERE / name).write_bytes(encoded)
    return {"file": name, "trace_sha256": sha256(encoded).hexdigest(), "events": dict(sorted(status.items()))}

if __name__ == "__main__":
    assert len(sys.argv) == 3
    export.write("chronology.json", {mode: render(mode, Path(path).resolve()) for mode, path in zip(ARMS, sys.argv[1:])})
