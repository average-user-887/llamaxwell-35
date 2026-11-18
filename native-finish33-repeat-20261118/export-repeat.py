#!/usr/bin/env python3
"""Verify owner-held inputs and export the counterbalanced repeat without private data."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REFERENCE_NAME = "native-finish33-sampler-model-20261118"
STABLE = CAMPAIGN / "github-worktree" / REFERENCE_NAME
if not STABLE.is_dir():
    STABLE = HERE.parent / REFERENCE_NAME
sys.path.insert(0, str(STABLE))

import export as base
from trace_export import annotated, phases, read_text

MODES = ("serial", "native")
PHASES = ("cold", "warm")
COUNTERS = {"row0", "row1", "combined", "skip", "q2_sparse", "q2_stride"}
SETTINGS = ("prompt_n", "n_predict", "seed", "temperature", "cache_prompt", "ignore_eos")
TIMINGS = ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
           "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
           "draft_n", "draft_n_accepted")
CLOCK = re.compile(r"^(\d+\.\d{2}\.\d{3}\.\d{3}) [A-Z] ")
READY = re.compile(r"^owned server ready after (\d+)s$", re.M)
HEALTH_TIMEOUT = re.compile(r"^owned server failed to become healthy within (\d+)s$", re.M)
FAILURES = (
    ("owned_worker_startup_exit", "owned server exited or could not be inspected during startup"),
    ("owned_endpoint_bind", "owned loopback endpoint unavailable"),
    *base.FAILURES,
)
SOFTWARE = {"exe", "libllama-server-impl.so", "libggml-cuda.so", "libllama.so.0.6.0",
            "libllama-common.so.0.6.0"}


def write(name, content):
    target = HERE / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(target)


def write_json(name, value):
    write(name, json.dumps(value, indent=2, allow_nan=False) + "\n")


def numbers(record, keys):
    return {key: value for key in keys if type(value := record.get(key)) in (bool, int, float)
            and math.isfinite(value)}


def failure_metadata(raw, official, validation):
    log = read_text(raw / "run.log")
    ready, timeout = READY.search(log), HEALTH_TIMEOUT.search(log)
    runner, cleanup = (official[key] for key in ("runner_exit", "cleanup_exit"))
    known = next((kind for kind, fragment in FAILURES if fragment in log), None)
    fatal = ("owned_worker_health_timeout" if timeout else known or
             ("arm_deadline" if runner == 124 else "runner_failure")) if runner else None
    return {
        "runner_exit": runner,
        "cleanup_exit": cleanup,
        "runner_failure_class": fatal,
        "cleanup_outcome": "failed" if cleanup else "passed",
        "validation_disposition": validation,
        "owned_worker_ready_s": int(ready[1]) if ready else None,
        "health_check_limit_s": int(timeout[1]) if timeout else None,
    }


def inspect(raw, mode, original=False):
    expected_owners = {CAMPAIGN / "finish33-sampler-r1"}
    if original:
        expected_owners.add(CAMPAIGN / "finish33-sampler")
    assert raw.parent.name == "receipts" and raw.parent.parent in expected_owners
    base.verify_observations(raw)
    official = base.read(raw / "sanitized/model-receipt.json")
    status = {key: int(value) for key, value in base.STATUS.findall((raw / "status.txt").read_text())}
    manifest = {key: int(value) for key, value in base.START.findall((raw / "manifest.txt").read_text())}
    assert mode == official["mode"]
    assert (raw / "manifest.txt").read_text().splitlines()[0] == f"stage={mode}"
    assert all(status[key] == official[key] for key in ("runner_exit", "cleanup_exit"))
    assert manifest["start_epoch"] <= status["completed_epoch"]
    assert manifest["deadline_epoch"] == status["arm_deadline_epoch"]
    validated, disposition = None, "terminal_failure"
    if not status["cleanup_exit"] and (not status["runner_exit"] or original):
        checked = base.receipt(raw)
        validated = checked["validated"]
        disposition = checked["validation_disposition"]
    responses = {phase: base.read(raw / f"{phase}.json") for phase in PHASES
                 if (raw / f"{phase}.json").is_file()}
    snapshots = {key: numbers(value, COUNTERS) for key, value in official["snapshots"].items()
                 if key in {"start", *PHASES}}
    for counters in snapshots.values():
        assert set(counters) == COUNTERS and all(type(v) is int and v >= 0 for v in counters.values())
    if validated:
        assert set(responses) == set(PHASES) and set(snapshots) == {"start", *PHASES}
        assert validated["technical_pass"] and validated["semantic_score"] is None
        assert set(validated["actual_server_software_sha256"]) == SOFTWARE
        assert all(re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in validated["actual_server_software_sha256"].values())
    return {
        "mode": mode, "status": {**manifest, **status},
        "terminal": failure_metadata(raw, official, disposition),
        "validated": validated, "responses": responses, "snapshots": snapshots,
    }


def trace(raw, label, info):
    mode = info["mode"]
    responses, snapshots = (info[key] for key in ("responses", "snapshots"))
    log_segments, issue = phases(raw, False)
    source_lines = {phase: part.splitlines() for phase, part in log_segments.items()}
    events = [{"event": "owned_run_begin", "arm": mode, "start_epoch": info["status"]["start_epoch"]}]
    ready = info["terminal"]["owned_worker_ready_s"]
    if ready is not None:
        events.append({"event": "owned_health_observed", "arm": mode, "elapsed_s": ready})

    def physical(phase):
        return {"event": "physical_counter_boundary", "arm": mode, "boundary": phase,
                "counters": snapshots[phase]}

    for row in annotated(mode, raw, int(mode == "native")):
        event, phase = row["event"], row.get("phase")
        if event == "request_begin" and phase == "cold" and "start" in snapshots:
            events.append(physical("start"))
        if "relative_log_line" in row:
            line = source_lines[phase][row["relative_log_line"] - 1]
            if started := CLOCK.match(line):
                row["process_log_clock"] = started[1]
        if event == "log_discontinuity" and ready is None and not responses and log_segments:
            row["event"] = "startup_log_only_no_healthy_boundary"
            issue = row["event"]
        if event == "output_boundary" and phase in responses:
            record = responses[phase]
            if mode == "native":
                # The response carries every emitted ID. The ordinal is public, while the
                # actual ID and any timing or cycle association absent from the logs stay private.
                for ordinal, _ in enumerate(record["token_ids"], 1):
                    events.append({"event": "observed_stream_output_ordinal", "arm": mode,
                                   "phase": phase, "output_token": ordinal})
        events.append(row)
        if event == "output_boundary" and phase in responses:
            events.append({"event": "observed_response_metrics", "arm": mode, "phase": phase,
                           "request": numbers(record, SETTINGS),
                           "client": numbers(record, ("ttft_s", "wall_s")),
                           "server_timings": numbers(record["timings"], TIMINGS)})
            if phase in snapshots:
                events.append(physical(phase))
    terminal = info["terminal"]
    events.append({"event": "owned_run_outcome", "arm": mode, **terminal, "status": info["status"]})
    counts = Counter(row["event"] for row in events)
    if info["validated"]:
        log = read_text(raw / "server.log")
        labels = {
            "host_counter_snapshot": "mv1_host: rounds=",
            "fast_selector_first_dispatch": "q2_r2_top16_dispatch",
            "tiled_probe_first_dispatch": "q2_r2_probe4_dispatch",
            "fixed_stride_first_dispatch": "q2_r2_stride_dispatch",
            "sampler_snapshot_completion": "sampler_snapshot:",
            "sparse_attention_dispatch_sample": "ggml_cuda: occamy_sparse_probe dispatch calls=",
            "q2_moe_residency": "Q2_COHORT residency",
            "recurrent_rollback_rows": "llama_context: n_rs_seq",
        }
        for event, literal in labels.items():
            assert counts[event] == log.count(literal), event
        assert issue is None and counts["host_counter_record_incomplete"] == 0
        assert counts["fixed_stride_first_dispatch"] == int(mode == "native")
        assert counts["offload_observed"] == 1 and counts["sampler_snapshot_completion"] == 2
        assert counts["output_boundary"] == 2 and counts["physical_counter_boundary"] == 3
        for phase, record in responses.items():
            stage_events = [row for row in events if row.get("phase") == phase]
            assert sum(row["event"] == "sse_clock_checkpoint" for row in stage_events) == len(
                record.get("observed_sse_checkpoints", []))
            output = next(row for row in stage_events if row["event"] == "output_boundary")
            assert output["observed_output_count"] == len(record["token_ids"])
            if mode == "serial":
                assert sum(row["emitted"] for row in stage_events if row["event"] == "cycle") == len(record["token_ids"])
            else:
                assert sum(row["event"] == "observed_stream_output_ordinal" for row in stage_events) == len(record["token_ids"])
    filename = f"{label}.trace-v2.jsonl"
    encoded = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in events)
    write(filename, encoded)
    return {"file": filename, "trace_sha256": sha256(encoded.encode()).hexdigest(),
            "events": dict(sorted(counts.items())), "log_boundary_issue": issue,
            "response_phases_observed": [phase for phase in PHASES if phase in responses],
            "start_epoch": info["status"]["start_epoch"],
            "completed_epoch": info["status"]["completed_epoch"],
            "terminal": terminal}


def clean(info):
    terminal = info["terminal"]
    return (terminal["runner_exit"], terminal["cleanup_exit"]) == (0, 0) and bool(info["validated"])


def identity(paths, observations):
    software = [info["validated"]["actual_server_software_sha256"] for info in observations]
    assert software.count(software[0]) == 4
    pinned = base.read(STABLE / "comparison.json")
    assert software[0]["libggml-cuda.so"] == pinned["candidate_cuda_sha256"]
    assert software[0]["libllama-server-impl.so"] == pinned["candidate_server_impl_sha256"]
    vectors = [(sorted((raw / "child-selected-environment.txt").read_text().splitlines()),
                (raw / "child-server.args").read_text().splitlines()) for raw in paths]
    assert all(env == vectors[0][0] for env, _ in vectors)
    assert vectors[0][1] == vectors[2][1] and vectors[1][1] == vectors[3][1]
    serial, native = vectors[0][1], vectors[1][1]
    spec = serial.index("--spec-type")
    assert serial[:spec] == native[:spec]
    assert serial[spec:] == ["--spec-type", "none"]
    assert native[spec:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
                             "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16"]
    return {"all_mapped_software_matches": software[0], "shared_argv_differs_only_in_speculation": True,
            "all_environment_vectors_equal": True, "serial_speculation_args": serial[spec:],
            "native_speculation_args": native[spec:],
            "treatment_flags": base.read(STABLE / "runtime-pair.json")["observed_enabled_treatment_flags"]}


def measure(records):
    serial, native = records
    sr, nr = (1000 * len(row["token_ids"]) / row["timings"]["predicted_ms"] for row in records)
    wall_serial, wall_native = (row["wall_s"] for row in records)
    rounds = native["timings"]["draft_n"]
    count = len(native["token_ids"])
    elapsed_ms = native["timings"]["predicted_ms"]
    return {"serial_decode_tok_s": sr, "native_decode_tok_s": nr, "decode_gain_pct": 100 * (nr / sr - 1),
            "meets_25_pct": nr >= sr * 1.25, "serial_wall_s": wall_serial, "native_wall_s": wall_native,
            "native_wall_saved_s": wall_serial - wall_native, "output_per_offer": count / rounds,
            "native_ms_per_offer": elapsed_ms / rounds, "budget_ms_per_offer_at_100": 10 * count / rounds,
            "additional_ms_per_offer_to_100": max(0, (elapsed_ms - 10 * count) / rounds),
            "native_offered": rounds, "native_accepted": native["timings"]["draft_n_accepted"]}


def compare(paths, observations):
    serial, native, repeat_serial, repeat_native = observations
    slots = (native, serial, repeat_serial, repeat_native)
    assert len(set(paths)) == 4
    assert all(info["status"]["completed_epoch"] <= next_info["status"]["start_epoch"]
               for info, next_info in zip(slots, slots[1:]))
    containers = [next(row for row in (path / "status.txt").read_text().splitlines() if row.startswith("container="))
                  for path in paths]
    assert len(set(containers)) == 4
    assert all(clean(info) for info in observations[2:])
    report = {"original_pair_order": ["native", "serial"], "repeat_pair_order": ["serial", "native"],
              "arms_used_four_distinct_owned_workers": True, "arms_did_not_overlap": True,
              "same_phase_settings_and_private_prompt_tokens": True,
              "runtime": identity(paths, observations),
              "semantic_score": None, "phases": {},
              "original_terminal": {info["mode"]: info["terminal"] for info in observations[:2]}}
    for phase in PHASES:
        records = [info["responses"][phase] for info in observations]
        os, on, rs, rn = records
        keys = (*SETTINGS, "prompt_sha256")
        assert all(tuple(row[key] for key in keys) == tuple(os[key] for key in keys) for row in records)
        assert all(row["timings"]["cache_n"] == os["timings"]["cache_n"] and
                   row["timings"]["prompt_n"] == os["timings"]["prompt_n"] for row in records)
        expected_count = {"cold": 192, "warm": 3072}[phase]
        assert all(len(row["token_ids"]) == row["timings"]["predicted_n"] == expected_count for row in records)
        parity = {key: base.parity(left, right) for key, left, right in (
            ("original_serial_vs_repeat_serial", os, rs), ("original_native_vs_repeat_native", on, rn),
            ("original_serial_vs_native", os, on), ("repeat_serial_vs_native", rs, rn))}
        for key in ("original_serial_vs_repeat_serial", "original_native_vs_repeat_native"):
            checks = parity[key]
            assert checks["first_id_difference_1_based"] is None and checks["full_text_equal"]
            assert checks["repetition_metrics_equal"] and checks["same_proposal_and_acceptance_totals"]
        assert parity["original_serial_vs_native"] == parity["repeat_serial_vs_native"]
        old, new = measure((os, on)), measure((rs, rn))
        report["phases"][phase] = {
            "request": numbers(rs, SETTINGS), "prompt_reused": rs["timings"]["cache_n"],
            "prompt_evaluated": rs["timings"]["prompt_n"],
            "original_pair": old, "repeat_pair": new,
            "both_pairs_meet_25_pct": old["meets_25_pct"] and new["meets_25_pct"],
            "paired_decode_gain_pct_spread_points": abs(old["decode_gain_pct"] - new["decode_gain_pct"]),
            "native_decode_tok_s_spread": abs(old["native_decode_tok_s"] - new["native_decode_tok_s"]),
            "serial_decode_tok_s_spread": abs(old["serial_decode_tok_s"] - new["serial_decode_tok_s"]),
            "output_parity": parity,
            "normalized_repetition": {key: base.repetition(row["text"]) for key, row in zip(
                ("original_serial", "original_native", "repeat_serial", "repeat_native"), records, strict=True)},
        }
    old_host, repeat_host = (base.host_points(path) for path in (paths[1], paths[3]))
    matched = sorted(old_host.keys() & repeat_host.keys())
    first_difference = next((i for i in matched if old_host[i][:2] != repeat_host[i][:2]), None)
    report["native_warm_host_window"] = (
        base.host_comparison(paths[1], paths[3]) if first_difference is None else
        {"window_available": False, "matched_snapshots": len(matched),
         "first_cumulative_acceptance_or_restore_difference_round": first_difference})
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--failed-serial", action="append", default=[], type=Path)
    cli.add_argument("--failed-native", action="append", default=[], type=Path)
    cli.add_argument("--repeat", nargs=4, type=Path, metavar="RECEIPT",
                     help="original serial, original native, repeat serial, repeat native")
    args = cli.parse_args()
    attempts = {}
    failed_receipts = []
    for mode in MODES:
        for index, raw in enumerate(getattr(args, "failed_" + mode), 1):
            raw = raw.resolve()
            info = inspect(raw, mode)
            assert not clean(info)
            failed_receipts.append(info)
            label = f"failed-{mode}-{index}"
            attempts[label] = trace(raw, label, info)
    if args.repeat:
        paths = [raw.resolve() for raw in args.repeat]
        observations = [inspect(raw, mode, original=index < 2)
                        for index, (raw, mode) in enumerate(zip(paths, MODES * 2, strict=True))]
        for raw, info in zip(paths[2:], observations[2:], strict=True):
            label = "repeat-" + info["mode"]
            attempts[label] = trace(raw, label, info)
    if not attempts:
        cli.error("supply one or more terminal input receipts")
    write_json("attempts.json", attempts)
    if args.repeat:
        comparison = compare(paths, observations)
        completed_original = observations[0]["status"]["completed_epoch"]
        for info in failed_receipts:
            assert completed_original <= info["status"]["start_epoch"]
            assert info["status"]["completed_epoch"] <= observations[2 + MODES.index(info["mode"])]["status"]["start_epoch"]
        write_json("repeat.report.json", comparison)


if __name__ == "__main__":
    main()
