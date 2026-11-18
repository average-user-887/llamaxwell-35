"""Export complete typed chronology for the combined candidate's two original model arms."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import export
from trace_export import annotated, read_text

KEYS = {"row0", "row1", "combined", "skip", "q2_sparse", "q2_stride"}
ARMS = ("serial", "native")

def render(mode, raw):
    export.verify_observations(raw)
    receipt = export.receipt(raw)
    assert receipt["mode"] == mode
    passed = bool(receipt["validated"] and receipt["validated"]["technical_pass"])
    assert passed and receipt["cleanup_exit"] == 0
    snapshots = receipt["read_only_counter_snapshots"]
    assert set(snapshots) <= {"start", "cold", "warm"}
    if passed:
        assert set(snapshots) == {"start", "cold", "warm"}
    for words in snapshots.values():
        assert set(words) == KEYS and all(type(v) is int and v >= 0 for v in words.values())
    def boundary(label):
        return {"event": "physical_counter_boundary", "arm": mode, "boundary": label, "counters": snapshots[label]}
    events, status = [], Counter()
    for row in annotated(mode, raw, int(mode == "native")):
        if row["event"] == "request_begin" and row["phase"] == "cold" and "start" in snapshots:
            events.append(boundary("start"))
        events.append(row)
        if row["event"] == "output_boundary" and row["phase"] in receipt["request"]:
            phase = row["phase"]
            metrics = receipt["request"][phase]
            keys = ("ttft_s", "server_prompt_ms", "server_decode_ms", "offered_drafts", "accepted_drafts")
            events.append({"event": "observed_response_metrics", "arm": mode, "phase": phase,
                           **{key: metrics[key] for key in keys}})
            if phase in snapshots:
                events.append(boundary(phase))
    events.append({"event": "owned_run_outcome", "arm": mode, "runner_exit": receipt["runner_exit"],
                   "cleanup_exit": receipt["cleanup_exit"], "duration_s": receipt["bounded_wall_s"],
                   "raw_failure_class": receipt["raw_failure_class"], "status": receipt["typed_status"],
                   "validation_disposition": receipt["validation_disposition"]})
    status.update(row["event"] for row in events)
    if passed:
        raw_log = read_text(raw / "server.log")
        literal_events = {
            "host_counter_snapshot": "mv1_host: rounds=",
            "fast_selector_first_dispatch": "q2_r2_top16_dispatch",
            "tiled_probe_first_dispatch": "q2_r2_probe4_dispatch",
            "fixed_stride_first_dispatch": "q2_r2_stride_dispatch",
            "sampler_snapshot_completion": "sampler_snapshot:",
            "sparse_attention_dispatch_sample": "ggml_cuda: occamy_sparse_probe dispatch calls=",
            "q2_moe_residency": "Q2_COHORT residency",
            "recurrent_rollback_rows": "llama_context: n_rs_seq",
        }
        for kind, label in literal_events.items():
            assert status[kind] == raw_log.count(label), (kind, status[kind], raw_log.count(label))
        assert status["host_counter_record_incomplete"] == 0 and status["offload_observed"] == 1
        assert status["fixed_stride_first_dispatch"] == int(mode == "native")
        assert status["sampler_snapshot_completion"] == 2
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
    if len(sys.argv) == 4 and sys.argv[1] == "--arm" and sys.argv[2] in ARMS:
        mode = sys.argv[2]
        export.write(f"{mode}.chronology.json", render(mode, Path(sys.argv[3]).resolve()))
    elif len(sys.argv) == 3:
        export.write("chronology.json", {mode: render(mode, Path(path).resolve()) for mode, path in zip(ARMS, sys.argv[1:])})
    else:
        raise SystemExit("usage: chronology.py SERIAL_RECEIPT NATIVE_RECEIPT | --arm MODE RECEIPT")
