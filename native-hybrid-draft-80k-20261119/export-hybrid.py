#!/usr/bin/env python3
"""Verify and export safe typed events from the two original hybrid receipts."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parents[1]
REPEAT = ROOT.parent / "native-finish33-repeat-20261118/export-repeat.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


repeat = load(REPEAT, "repeat_export")
base = repeat.base
control = load(CAMPAIGN / "moon100-hybrid-serial/model.py", "paired_control")
repeat.HERE = ROOT


def inspect(raw, mode):
    base.verify_observations(raw)
    official = base.read(raw / "sanitized/model-receipt.json")
    status = {key: int(value) for key, value in base.STATUS.findall((raw / "status.txt").read_text())}
    manifest = {key: int(value) for key, value in base.START.findall((raw / "manifest.txt").read_text())}
    assert (raw / "manifest.txt").read_text().splitlines()[0] == f"stage={mode}"
    assert mode == official["mode"]
    assert all(status[key] == official[key] for key in ("runner_exit", "cleanup_exit"))
    assert manifest["start_epoch"] <= status["completed_epoch"] <= manifest["deadline_epoch"]
    assert manifest["deadline_epoch"] == status["arm_deadline_epoch"]
    ready = not (official["runner_exit"] or official["cleanup_exit"])
    validated = None
    disposition = "terminal_failure"
    if ready:
        if mode == "native":
            control.hybrid.result_comparison(raw)
            validated = control.hybrid.collect("native", raw)
        else:
            validated = control.finished(raw)
        assert official["validated"] == validated
        disposition = "passed_original_terminal_validator"

    responses = {phase: base.read(raw / f"{phase}.json") for phase in repeat.PHASES
                 if (raw / f"{phase}.json").is_file()}
    snapshots = {key: repeat.numbers(value, repeat.COUNTERS) for key, value in official["snapshots"].items()
                 if key in {"start", *repeat.PHASES}}
    assert all(set(row) == repeat.COUNTERS for row in snapshots.values())
    terminal = repeat.failure_metadata(raw, official, disposition)
    refusal = (raw / "run.log").read_text().splitlines()
    for title, message in (
        ("serial_control_exact_fixture", "archived exact fixture changed"),
        ("serial_control_exact_output", "serial IDs or text differ from archived control"),
        ("serial_control_draft_marker", "no-MTP control produced draft graph evidence"),
        ("serial_control_physical_marker", "warm serial q1 or same-slot cache restore was not observed"),
    ):
        if not ready and any(message in line for line in refusal):
            terminal["runner_failure_class"] = title
    return {"mode": mode, "status": {**manifest, **status}, "terminal": terminal,
            "validated": validated, "responses": responses, "snapshots": snapshots}


MERGE = re.compile(r"^(\d+\.\d{2}\.\d{3}\.\d{3}) I spec common_specu: MTP merge re-eval enabled: "
                   r"verify re-eval folded into the next draft step$")
GRAPH = re.compile(r"^(\d+\.\d{2}\.\d{3}\.\d{3}) I mtp_last_graph pos=(\d+) rows=(\d+) outputs=(\d+)"
                   r" type=(\d+) kw=(\d+) vw=(\d+) qproj=(\d+) q=(\d+) nkv=(\d+) mask=(\d+) ffn=(\d+) ordered=(\d+)$")
GRAPH_FIELDS = ("position", "draft_rows", "output_rows", "compute_graph_type", "key_write_rows",
                "value_write_rows", "query_projection_rows", "query_rows", "key_width", "mask_rows",
                "ffn_rows", "ordered_cache_writes")


def trace(raw, info):
    mode = info["mode"]
    label = "hybrid-native" if mode == "native" else "hybrid-serial"
    meta = repeat.trace(raw, label, info)
    dest = ROOT / meta["file"]
    events = [json.loads(line) for line in dest.read_text().splitlines()]
    logs, _ = repeat.phases(raw, False)
    extra = []
    for phase, log in logs.items():
        for number, line in enumerate(log.splitlines(), 1):
            if match := MERGE.fullmatch(line):
                extra.append({"event": "merged_draft_catchup_enabled", "arm": mode, "phase": phase,
                              "relative_log_line": number, "process_log_clock": match[1]})
            if match := GRAPH.fullmatch(line):
                extra.append({"event": "draft_final_query_first_dispatch", "arm": mode, "phase": phase,
                              "relative_log_line": number, "process_log_clock": match[1],
                              **dict(zip(GRAPH_FIELDS, map(int, match.groups()[1:]), strict=True))})
    if mode == "native":
        assert [(row["event"], row["phase"]) for row in extra] == [
            ("merged_draft_catchup_enabled", "startup"), ("draft_final_query_first_dispatch", "cold")]
    else:
        assert not extra
    for row in extra:
        phase = row["phase"]
        index = next((i for i, event in enumerate(events)
                      if event.get("phase") == phase and (
                        event.get("relative_log_line", 0) > row["relative_log_line"] or
                        event["event"] == "output_boundary")), None)
        if index is None and phase == "startup":
            index = next(i for i, event in enumerate(events) if event["event"] == "request_begin")
        assert index is not None
        events.insert(index, row)
    data = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in events)
    dest.write_text(data)
    meta["events"] = dict(sorted(Counter(row["event"] for row in events).items()))
    meta["trace_sha256"] = repeat.sha256(data.encode()).hexdigest()
    return meta


def repetition_audit(serial, native):
    result = {"normalization": "collapse whitespace and replace digit runs; count excess occurrences",
              "semantic_score": None, "phases": {}}
    archives = {"clean_split_reference": control.hybrid.baseline, "clean_serial_reference": control.SERIAL_REF}
    for phase in repeat.PHASES:
        records = {key: base.read(path / f"{phase}.json") for key, path in archives.items()}
        if phase in native["responses"]:
            records["hybrid_native"] = native["responses"][phase]
        if phase in serial["responses"]:
            records["hybrid_serial"] = serial["responses"][phase]
        pairs = (("hybrid_native", "clean_split_reference"),
                 ("hybrid_serial", "clean_serial_reference"),
                 ("hybrid_native", "hybrid_serial"))
        comparisons = {}
        for left, right in pairs:
            if left in records and right in records:
                parity = base.parity(records[left], records[right])
                if right.endswith("_reference") and (native if left.endswith("native") else serial)["validated"]:
                    assert parity["first_id_difference_1_based"] is None
                    assert parity["full_text_equal"] and parity["repetition_metrics_equal"]
                    assert parity["same_proposal_and_acceptance_totals"]
                comparisons[f"{left}_vs_{right}"] = parity
        result["phases"][phase] = {"records": {name: base.repetition(record["text"])
                                               for name, record in records.items()}, "parity": comparisons}
    return result


def identity(serial, native, pair):
    assert pair["actual_same_build_runtime_flags"]
    vectors = [(sorted((raw / "child-selected-environment.txt").read_text().splitlines()),
                (raw / "child-server.args").read_text().splitlines()) for raw in (serial, native)]
    (env, serial_args), (env_native, native_args) = vectors
    assert env == env_native
    index = serial_args.index("--spec-type")
    assert serial_args[:index] == native_args[:index]
    assert serial_args[index:] == ["--spec-type", "none"]
    assert native_args[index:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
                                   "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16"]
    retained = {"GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16=1", "GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4=1",
                "GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE=1", "LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE=1",
                "LLAMA_MTP_MERGE_REEVAL=1", "LLAMA_MTP_LAST_QUERY=1"}
    assert retained <= set(env)
    return {"same_observed_server_software": pair["actual_same_build_mapped_software"],
            "same_child_environment": True, "only_argv_difference_is_speculation": True,
            "native_speculation_args": native_args[index:], "serial_speculation_args": serial_args[index:],
            "common_enabled_flags": sorted(retained)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    args = parser.parse_args()
    serial_path = args.serial.resolve()
    assert serial_path.parent == control.HERE / "receipts"
    native_path = control.NATIVE
    serial, native = (inspect(path, mode) for path, mode in ((serial_path, "serial"), (native_path, "native")))
    assert native["status"]["completed_epoch"] <= serial["status"]["start_epoch"]
    write("trace-index.json", {"arm_order": ["native", "serial"], "owned_arms_did_not_overlap": True,
                               "native": trace(native_path, native), "serial": trace(serial_path, serial)})
    write("repetition.json", repetition_audit(serial, native))
    if serial["validated"]:
        assert native["validated"]
        pair = control.compare(serial_path)
        metrics = {}
        for phase in repeat.PHASES:
            records = [info["responses"][phase] for info in (serial, native)]
            assert all(records[0][key] == records[1][key] for key in (*repeat.SETTINGS, "prompt_sha256"))
            metrics[phase] = repeat.measure(records)
            assert abs(pair["measurements"][phase]["same_build_native_over_serial_pct"]
                       - metrics[phase]["decode_gain_pct"]) < 1e-10
        write("same-build-pair.json", pair)
        write("pair-audit.json", {"same_private_prompt_and_settings": True,
                                  "implementation_identity": identity(serial_path, native_path, pair),
                                  "metrics": metrics, "semantic_score": None})
    elif (ROOT / "same-build-pair.json").exists() or (ROOT / "pair-audit.json").exists():
        raise ValueError("failure replay would leave older successful-pair exports")
    print("PASS: original receipts verified and all safe traces and repetition metrics exported")

if __name__ == "__main__":
    main()
