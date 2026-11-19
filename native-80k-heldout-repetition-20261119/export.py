"""Revalidate sealed local measurements and export only typed numerical evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess

COUNTERS = ("row0", "row1", "combined", "skip", "q2_sparse", "q2_stride")
CASES = ("n32", "n72")
SAFE_CASE_FIELDS = (
    "fixture", "request_params", "actual_sampling", "prompt_tokens", "cached_prompt_tokens",
    "completion_tokens_including_eos", "stop_type", "draft_offered", "draft_accepted",
    "server_prompt_ms", "server_decode_ms", "start_epoch_ns", "end_epoch_ns", "wall_s",
    "quality", "archive_parity", "repetition", "expected_source_repetition",
)


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify(out):
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", "observations.sha256"], cwd=out, check=True)


def safe_case(row):
    assert set(row) == {*SAFE_CASE_FIELDS, "prompt_sha256", "template_sha256"}
    return {key: row[key] for key in SAFE_CASE_FIELDS}


def runtime(out, labels):
    status = dict(re.findall(r"^(runner_exit|cleanup_exit|completed_epoch)=(\d+)$",
                             (out / "status.txt").read_text(), re.M))
    manifest = dict(re.findall(r"^(start_epoch|deadline_epoch|cap_seconds)=(\d+)$",
                               (out / "manifest.txt").read_text(), re.M))
    image, = re.findall(r"^image=sha256:([a-f0-9]{64})$", (out / "manifest.txt").read_text(), re.M)
    fields = {k: int(v) for k, v in {**manifest, **status}.items()}
    assert set(fields) == {"start_epoch", "deadline_epoch", "cap_seconds", "completed_epoch", "runner_exit", "cleanup_exit"}
    flags = dict(re.findall(r"^((?:GGML_CUDA|LLAMA)_[A-Z0-9_]+)=(-?\d+)$",
                            (out / "child-selected-environment.txt").read_text(), re.M))
    assert flags
    hardware, = re.findall(r"^M6000 telemetry age_s=([\d.]+) vram_used_MiB=(\d+) util_pct=(\d+) power_limit_W=(\d+)$",
                           (out / "admission.txt").read_text(), re.M)
    witness = {label: {key: read(out / "witness" / f"{label}.json")[key] for key in COUNTERS} for label in labels}
    return {
        **fields, "image_sha256": image, "effective_numeric_flags": {k: int(v) for k, v in flags.items()},
        "start_admission": dict(zip(("age_s", "vram_used_MiB", "util_pct", "power_limit_W"),
                                    (float(hardware[0]), *map(int, hardware[1:])))),
        "counter_witnesses": witness,
    }


def export(launcher, failed, serial, native, paired_input, output):
    for receipt in (failed, serial, native):
        verify(receipt)
    replayed = json.loads(subprocess.check_output(
        [launcher, "--compare", serial, native], stderr=subprocess.DEVNULL))
    assert replayed == read(paired_input)
    arms = {}
    bindings = set()
    for mode, folder in (("serial", serial), ("native", native)):
        local = read(folder / "sanitized/model-receipt.json")
        assert local["runner_exit"] == local["cleanup_exit"] == 0
        result = local["validated"]
        bindings.add(result["model_sha256"])
        public = {key: result[key] for key in (
            "technical_pass", "accepted_experts", "gpu_layers_offloaded", "server_context_tokens",
            "tokenizer_pre_override", "one_slot", "request_order", "actual_server_software_sha256", "route",
        )}
        assert result["cases"] == {case: replayed["cases"][case][mode] for case in CASES}
        arms[mode] = {**public, **runtime(folder, ("start", "cold", "warm")),
                      "bounded_wall_s": local["bounded_wall_s"]}
    refusal = read(failed / "sanitized/model-receipt.json")
    assert len(bindings) == 1
    assert refusal["runner_exit"] == 2 and refusal["cleanup_exit"] == 0
    assert refusal["validated"] is None and set(refusal["measurements"]) == {"n32"}
    assert "No owned-child counter sample for n32" in (failed / "run.log").read_text()
    assert {p.name for p in (failed / "witness").glob("*.json")} == {"start.json", "cold.json"}
    original = {
        **runtime(failed, ("start", "cold")), "bounded_wall_s": refusal["bounded_wall_s"], "mode": "serial",
        "original_observation_manifest_verified": True, "completed_request": safe_case(refusal["measurements"]["n32"]),
        "second_request_sent": False, "awaited_witness": "n32.json", "observed_witness": "cold.json",
        "failure": "The first post-request sample was written under the supervisor's fixed filename.",
        "correction": {"startup": "start.json", "n32": "cold.json", "n72": "warm.json"},
    }
    cases = {case: {
        "serial": safe_case(replayed["cases"][case]["serial"]),
        "native": safe_case(replayed["cases"][case]["native"]),
        "paired_parity": replayed["cases"][case]["paired_parity"],
    } for case in CASES}
    dump(output / "paired-scores.json", {"technical_pass": replayed["technical_pass"], "cases": cases})
    dump(output / "runtime.json", {"arms": arms, "same_preflight_model_binding": True,
                                   "benchmark_hardware": "Quadro M6000", "technique": "native MTP1",
                                   "phase_witness_roles": {"start": "startup", "cold": "after N32", "warm": "after N72"},
                                   "physical_scope": "Counter deltas are per request, not per-layer timings."})
    dump(output / "original-serial-failure.json", original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("launcher", "failed", "serial", "native", "paired_input", "output"):
        parser.add_argument(name, type=lambda value: Path(value).resolve())
    export(**vars(parser.parse_args()))
