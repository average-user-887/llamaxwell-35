#!/usr/bin/env python3
"""Read and compare the exact N88 Q8-enabled full-eight serial control."""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
REF = Path(os.environ["N88_PRIVATE_REFERENCE"])
spec = importlib.util.spec_from_file_location("q8_n88", C / "night-gdn-q8-long/long_probe.py")
n88 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n88)
n9, n18 = n88.n9, n88.n18
STAGES = ("cold", "warm")
SERVER_IMPL = C / "night-cache-only/libs/libllama-server-impl.so"


def settings():
    env, args = n88.arguments("on")
    return env, args[:args.index("--spec-type")] + ["--spec-type", "none"]


def check():
    n88.check_reference()
    n88.finished(REF)
    live = n88.inspect_perf("on", REF)
    if live != n9.read_json(REF / "result.json"):
        raise ValueError("archived native N88 Q8-on measurement differs")


def maps(path):
    rows = [row.split("  ", 1) for row in path.read_text().splitlines()]
    result = {key: digest for digest, key in rows}
    if len(result) != len(rows):
        raise ValueError("duplicated loaded runtime object")
    return result


def ready(out):
    env, args = settings()
    for filename, expected in (("environment.args", env), ("server.args", args),
                               ("pid1-server.args", ["/candidate/llama-server", *args])):
        if (out / filename).read_text().splitlines() != expected:
            raise ValueError("typed serial argv or profile differs")
    if sorted((out / "pid1-selected-environment.txt").read_text().splitlines()) != sorted(env[1::2]):
        raise ValueError("actual Q8 serial flags or instrumentation differ")
    expected = {**maps(REF / "runtime-library-hashes.txt"), str(SERVER_IMPL): n9.hash_file(SERVER_IMPL)}
    observed = maps(out / "runtime-library-hashes.txt")
    if observed != expected:
        raise ValueError("actual serial implementation, executable or native maps differ")
    start = (out / "server-start.log").read_text()
    for name, values in (("n_ctx", ["84224"]), ("n_seq_max", ["1"]), ("n_ubatch", ["1024"]),
                         ("n_rs_seq", ["0"]), ("n_outputs_max_per_seq", ["1"])):
        if re.findall(rf"llama_context: {name}\s+=\s*(\d+)\b", start) != values:
            raise ValueError(f"actual serial {name} differs")
    if (any(marker not in start for marker in n9.START_MARKERS) or n88.short.CUDA_ERROR.search(start) or
            not all(set(re.findall(pattern, start, re.M)) == {"8"} for pattern in n88.n48.WIDTHS) or
            "creating MTP draft context" in start):
        raise ValueError("effective 42/42, eight-expert or no-MTP startup gate failed")
    n9.require_admission(out / "admission.txt")
    return start, observed


def collect(out):
    start, mapping = ready(out)
    pre, full = [(out / name).read_text() for name in ("server-pre-warm.log", "server.log")]
    if (not pre.startswith(start) or not full.startswith(pre) or n88.short.CUDA_ERROR.search(full) or
            n18.SHIFT.search(full) or n9.CACHE_MARKER.search(full) or n9.FULL_MARKER.search(full) or
            re.search(r"accepted\s+\d+/\s*\d+ draft tokens|statistics\s+draft-mtp:", full) or n88.COUNTER.search(full)):
        raise ValueError("serial log discontinuity, state shift, CUDA fault or nonserial verifier evidence")
    warm = full[len(pre):]
    if n88.n48.sparse_shapes(warm) != [1] or "n_past = 79996" not in warm:
        raise ValueError("warm serial q1 or same-slot cache restore was not observed")
    fixture = n88.short.fixture("80k")
    result = {}
    for phase, log in zip(STAGES, (pre[len(start):], warm)):
        data = n9.read_json(out / f"{phase}.json")
        reference = n9.read_json(REF / f"{phase}.json")
        numeric, errors = (n9.check_record(n9.CONTROL, phase, data, reference, fixture) if phase == "cold"
                           else n18.warm_summary(n18.CONTROL, data, reference, fixture))
        if errors or len(re.findall(r"stop processing: n_tokens = \d+, truncated = 0\b", log)) != 1:
            raise ValueError("serial request has incomplete output, effective cache or timing")
        result[phase] = {key: numeric[key] for key in
                         ("cache_n", "evaluated_prompt_n", "n_out", "ttft_s", "request_wall_s", "decode_s", "decode_toks_s")}
    return {"technical_pass": True, "semantic_score": None, "q8a_flag_on": True, "warm_serial_sparse_widths": [1],
            "q8a_t2_launches_observed": 0, "metrics": result,
            "mapped_runtime": {Path(name).name: value for name, value in mapping.items()}}


def compare(out):
    n88.finished(out)
    current = collect(out)
    if current != n9.read_json(out / "result.json"):
        raise ValueError("archived serial receipt failed replay")
    archived = n88.inspect_perf("on", REF)
    comparisons = {}
    for phase in STAGES:
        a, b = current["metrics"][phase], archived["metrics"][phase]
        x, y = (n9.read_json(path / f"{phase}.json") for path in (out, REF))
        comparisons[phase] = {
            "serial": a, "native_decode_toks_s": b["decode_toks_s"], "native_request_wall_s": b["request_wall_s"],
            "native_over_serial_decode": b["decode_toks_s"] / a["decode_toks_s"],
            "serial_minus_native_wall_s": a["request_wall_s"] - b["request_wall_s"],
            "first_id_difference_1_based": n9.first_mismatch(x["token_ids"], y["token_ids"]),
            "text_equal": x["text"] == y["text"],
        }
    return {"scope": "same N88 Q8A-enabled runtime and full-eight profile with native depth one or no speculation",
            "semantic_score": None, "same_original_four_mapped_objects": True, "comparison": comparisons}


def finalize(out, runner, cleanup, seconds):
    safe = out / "sanitized"
    safe.mkdir()
    result = n9.read_json(out / "result.json") if runner == cleanup == 0 else None
    observations = {}
    for phase in STAGES:
        path = out / f"{phase}.json"
        if path.is_file():
            data = n9.read_json(path)
            observations[phase] = {key: data[key] for key in ("n_out", "wall_s", "ttft_s", "finish_reason")}
    n9.write_json(safe / "serial-receipt.json", {"runner_exit": runner, "cleanup_exit": cleanup,
                  "bounded_wall_s": seconds, "technical_pass": result is not None, "observations": observations,
                  "validated": result})


if __name__ == "__main__":
    try:
        action, *args = sys.argv[1:]
        if action == "check":
            check()
            print("Pinned N88 Q8A-on native receipt revalidated")
        elif action in ("env", "args"):
            print("\n".join(settings()[action == "args"]))
        elif action == "ready":
            ready(Path(args[0]))
        elif action == "verify":
            out = Path(args[0])
            n9.write_json(out / "result.json", collect(out))
        elif action == "compare":
            print(json.dumps(compare(Path(args[0])), indent=2))
        elif action == "finalize":
            finalize(Path(args[0]), *(int(value) for value in args[1:]))
        else:
            raise ValueError("unknown serial action")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"N88 serial: {exc}") from None
