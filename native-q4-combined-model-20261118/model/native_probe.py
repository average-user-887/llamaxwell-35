#!/usr/bin/env python3
"""Verify the same-build 80K serial and native Q4 R2 arms."""
import argparse
import csv
from observation import snapshot, attribution
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
sys.path.insert(0, str(C / "night-native-depth2-q3-80k-r1"))
import long_probe as original

n18 = original.n18
n9 = n18.n9
MODES = {"serial-on":0,"depth3-off":3,"depth3-attn":3,"depth3-combined":3,"smoke-off":3,"smoke-on":3}
ATTN_CUDA = "fd436e975b2729c6ccd96abf51ef3f5f054e73c734ce1efdc4141b996b245b6c"
Q4_CUDA = "abe7dcfdbf2d7df0b31b5478212f3234ffdd8e2b8f9a077c0a216b2e32865aae"
MOE_CUDA = "a68cf344d6c810cd8f9ca8bf46a7fcf8a0b2f466d660daa21ed576ef96da44b2"
OPERATOR = C / "resume-q4-r2-sparse/receipts/@PRIVATE_OPERATOR_RUN_NAME@"
GEOMETRY = re.compile(r"q4_r2_geometry nu=(\d+) threads=(\d+) nsplit=(\d+) blocks_per_sm=(\d+) n_kv=(\d+) nkvh=(\d+)")
PHASE = "LLAMA_MV3_PHASE=/native-phase.tsv"
REF = original.HERE / "receipts/@PRIVATE_BASELINE_RUN_NAME@"
SHA = {
    "result.json": "@PRIVATE_BASELINE_RESULT_SHA256@",
    "cold.json": "@PRIVATE_BASELINE_COLD_SHA256@",
    "warm.json": "@PRIVATE_BASELINE_WARM_SHA256@",
}
SERVER_IMPL = "1138515898887c6ec6be01a210cfb59bb8e1338ccf6eeef1d032ffd4fdb22b65"
HOST = re.compile(r"^mv1_host: rounds=(\d+) restores=(\d+)"
                  r" draft_ms=[\d.]+\(n=\d+\) tgt_decode_ms=[\d.]+\(n=\d+\)"
                  r" spec_process_ms=[\d.]+\(n=\d+\) sample_accept_ms=[\d.]+\(n=\d+\)"
                  r" acc_hist=\[\d+,\d+,\d+,\d+\]$", re.MULTILINE)
STAGES = ("cold", "warm")
FIXTURE = "@PRIVATE_PROMPT_FIXTURE_SHA256@"


def settings(mode, profile):
    env, args = original.settings("on")
    depth = MODES[mode]
    if depth:
        args[args.index("--spec-draft-n-max") + 1] = str(depth)
    else:
        args = args[:args.index("--spec-type")] + ["--spec-type", "none"]
    env += ["-e", f"GGML_CUDA_OCCAMY_FA_Q4_R2={int(mode in ('depth3-attn','depth3-combined','serial-on','smoke-on'))}",
            "-e", "GGML_CUDA_OCCAMY_FA_Q4_R2_NU=2", "-e", "GGML_CUDA_OCCAMY_FA_Q4_R2_CHECK=0"]
    env += ["-e",f"GGML_CUDA_OCCAMY_MOE_Q4_COHORTS={int(mode in ('depth3-combined','serial-on','smoke-on'))}",
            "-e","GGML_CUDA_OCCAMY_MOE_Q3_COHORTS=0","-e","LD_PRELOAD=/audit.so",
            "-e",f"NATIVE_Q4_PROMPT_N={2000 if mode.startswith('smoke') else 80000}"]
    if profile:
        env += ["-e", PHASE]
    return env, args


def qualify():
    for name, digest in SHA.items():
        if n9.hash_file(REF / name) != digest:
            raise ValueError("archived measured N100 r1 input changed")
    if not {"runner_exit=0", "cleanup_exit=0"} <= set((REF / "status.txt").read_text().splitlines()):
        raise ValueError("archived measured N100 r1 cleanup failed")
    if n9.read_json(REF / "result.json") != json.loads(json.dumps(original.collect("on", REF))):
        raise ValueError("archived measured N100 r1 failed receipt replay")
    if n9.fixture("80k")["prompt_sha256"] != FIXTURE:
        raise ValueError("80K fixture changed")
    operator = n9.read_json(OPERATOR / "result.json")
    build = n9.read_json(OPERATOR / "build.json")
    status = n9.key_values(OPERATOR / "status.txt")
    if (not operator["technical_pass"] or not operator["materiality_pass"] or operator["selected_nu"] != 2 or
            build["candidate_dso_sha256"] != ATTN_CUDA or
            (status["runner_exit"], status["cleanup_exit"]) != ("0", "0")):
        raise ValueError("Q4 NU2 operator or its CUDA lineage failed")


def check_moe():
    base=C/"resume-q4-moe-cohorts"
    physical,=[p for p in (base/"receipts").glob("*/result.json") if n9.read_json(p).get("technical_pass")]
    r=n9.read_json(physical)
    if (r["materiality_pass"] or not r["positive_generic_all_pairs"] or not r["incremental_30us_all_pairs"] or
        n9.hash_file(HERE/"libggml-cuda.so")!=Q4_CUDA or n9.hash_file(HERE/"moe-reference.so")!=MOE_CUDA or
        n9.hash_file(HERE/"attention-reference.so")!=ATTN_CUDA):
        raise ValueError("combined link or observed standalone technical/partial result differs")

def mapped(out):
    expected = {path: digest for digest, path in
                (line.split("  ", 1) for line in (original.N92 / "2/runtime-library-hashes.txt").read_text().splitlines())}
    del expected[str(C / "sparse/r2/libs/libggml-cuda.so")]
    expected["/candidate/libggml-cuda.so"] = Q4_CUDA
    expected[str(C / "night-cache-only/libs/libllama-server-impl.so")] = SERVER_IMPL
    expected["/audit.so"]=n9.hash_file(HERE/"audit.so")
    rows = [line.split("  ", 1) for line in (out / "runtime-library-hashes.txt").read_text().splitlines()]
    observed = {path: digest for digest, path in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("actual mapped target, head, Q4 CUDA or executable digest differs")
    return observed


def ready(mode, profile, out):
    env, args = settings(mode, profile)
    for name, expected in (("environment.args", env), ("server.args", args),
                           ("pid1-server.args", ["/candidate/llama-server", *args])):
        if (out / name).read_text().splitlines() != expected:
            raise ValueError(f"actual {name} differs from typed launch")
    expected = env[1::2]
    if sorted((out / "pid1-flags.txt").read_text().splitlines()) != sorted(expected):
        raise ValueError("actual native/phase flags differ or unexpected timer is enabled")
    snapshot(out/"counter-start.bin")
    maps = mapped(out)
    start = (out / "server-start.log").read_text(errors="replace")
    depth = MODES[mode]
    for name, expected in (("n_ctx", [84224]), ("n_seq_max", [1]), ("n_batch", [2048]), ("n_ubatch", [1024])):
        observed = list(map(int, re.findall(rf"llama_context: {name}\s+=\s*(\d+)", start)))
        if observed != expected * (2 if depth else 1):
            raise ValueError(f"actual {name} differs")
    markers = (*n9.START_MARKERS, "LLAMA_TRACE = 1")
    if any(s not in start for s in markers) or original.BAD.search(start):
        raise ValueError("42/42 GPU offload or one-slot trace missing")
    if re.findall(r"print_info: n_expert_used\s+=\s*(\d+)", start) != ["8"]:
        raise ValueError("effective eight-expert target profile differs")
    output_caps = re.findall(r"llama_context: n_outputs_max_per_seq\s+=\s*(\d+)", start)
    if output_caps != ([str(depth + 1), "1"] if depth else ["1"]):
        raise ValueError("effective target/draft output caps differ")
    caches = re.findall(r"llama_kv_cache: size = [^\n]*?K \((\w+)\): [^\n]*?V \((\w+)\):", start)
    if caches != [("f16", "f16")] * (2 if depth else 1):
        raise ValueError("effective F16 target/draft cache differs")
    creation = "creating MTP draft context" in start
    snapshots = re.findall(r"llama_context: n_rs_seq\s+=\s*(\d+)", start)
    expected = [str(depth), "0"] if depth else ["0"]
    if creation != bool(depth) or snapshots != expected:
        raise ValueError("actual native draft context or recurrent plane count differs")
    if depth and (f"n_max={depth}, n_min=0, p_min=0.00, n_embd=2048" not in start or
                  not re.search(r"nextn_predict_layers\s+u32\s+= 1\b", start) or
                  not re.search(rf"1 seqs\s+{depth} rs_seq\)", start)):
        raise ValueError("actual draft depth or single learned head differs")
    n9.require_admission(out / "admission.txt")
    return maps


def text_shape(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    return {"characters": len(text), "nonempty_lines": len(lines), "distinct_nonempty_lines": len(counts),
            "repeated_line_occurrences": len(lines) - len(counts),
            "largest_identical_line_count": max(counts.values(), default=0),
            "markdown_fence_lines": sum(line.startswith("```") or line.startswith("~~~") for line in lines)}


def cycles(mode, phase, log, emitted, complete):
    depth = MODES[mode]
    counts = original.n92.ACCEPT.findall(log)
    rows = []
    for i, (accepted, offered, restore) in enumerate(counts, 1):
        accepted, offered = int(accepted), int(offered)
        action = "checkpoint_restore_marker" if restore else ("native_trim_path" if accepted < offered else "all_accepted")
        rows.append({"arm": mode, "phase": phase, "cycle": i, "offered": offered, "accepted": accepted,
                     "emitted": None, "emission_provenance": "not_reconciled",
                     "rejected": offered - accepted, "state_action": action,
                     "target_ms": None, "draft_ms": None, "accept_ms": None, "cycle_ms": None,
                     "cost_provenance": "no_per_cycle_clock"})
    serial_estimate = emitted - sum(row["accepted"] + 1 for row in rows)
    if not depth:
        rows = [{"arm": mode, "phase": phase, "cycle": i, "offered": 0, "accepted": 0, "emitted": 1,
                 "rejected": 0, "state_action": "serial_output", "emission_provenance": "observed_stream_ID",
                 "target_ms": None, "draft_ms": None, "accept_ms": None, "cycle_ms": None,
                 "cost_provenance": "no_per_cycle_clock"} for i in range(1, emitted + 1)]
        return rows, 0
    if complete and not any(restore for _, _, restore in counts) and serial_estimate in (1, 2):
        for row in rows:
            row["emitted"] = row["accepted"] + 1
            row["emission_provenance"] = "native_accept_plus_one_reconciled_with_SSE_total"
        return rows, serial_estimate
    return rows, None


def log_phases(out):
    start, before, final = [(out / name).read_text(errors="replace") if (out / name).is_file() else ""
                            for name in ("server-start.log", "server-pre-warm.log", "server.log")]
    if before and before.startswith(start) and final.startswith(before):
        return start, before[len(start):], final[len(before):], final
    if not before and final.startswith(start):
        return start, final[len(start):], "", final
    raise ValueError("discontinuous or rotated server logs")


def phase_summary(path):
    with path.open() as f:
        output = []
        for row in csv.DictReader(f, delimiter="\t"):
            output.append({key: (int(row[key]) if key != "gpu_span_ms" else float(row[key])) for key in
                           ("ctx_type", "n_tokens", "calls", "synced", "gpu_span_ms")})
    if not output or not any(row["gpu_span_ms"] > 0 for row in output):
        raise ValueError("phase diagnostic has no measured GPU spans")
    return {"scope": "cumulative by context and batch width; includes startup/cold/warm, not matched to individual cycles",
            "rows": output}


def request(mode, phase, out, log):
    record = n9.read_json(out / f"{phase}.json")
    depth, count = MODES[mode], (192 if phase == "cold" else 3072)
    timings = record["timings"]
    old = n9.read_json(REF / f"{phase}.json")
    arm = n9.CANDIDATE if depth else n9.CONTROL
    base = n9.fixture("80k")
    result, errors = (n9.check_record(arm, phase, record, old, base) if phase == "cold"
                      else n18.warm_summary(arm, record, old, base))
    if errors:
        raise ValueError("; ".join(errors))
    if (record["n_out"] != count or not n9.elapsed(timings["prompt_ms"]) or
            len(re.findall(r"stop processing: n_tokens = \d+, truncated = 0\b", log)) != 1):
        raise ValueError("request finish or prompt evaluation not proven")
    if phase == "warm" and "n_past = 79996" not in log:
        raise ValueError("warm prompt-cache restore was not logged")
    cycle_rows, residual = cycles(mode, phase, log, count, True)
    if depth:
        histogram = original.n92.histogram(log, depth, timings["draft_n_accepted"], timings["draft_n"])
        if "statistics        draft-mtp:" not in log:
            raise ValueError("native draft-mtp runtime statistics missing")
        # Checkpoint replay changes the trace accounting; retain the raw attempt if it diverges.
        result.update(histogram)
        result["decode_ms_per_native_cycle"] = timings["predicted_ms"] / histogram["rounds"]
    elif original.n92.ACCEPT.search(log):
        raise ValueError("no-MTP arm emitted draft acceptance markers")
    result.update({"prompt_eval_s": timings["prompt_ms"] / 1000,
                   "unassigned_serial_outputs": residual, "structure": text_shape(record["text"]),
                   "draft_n_rejected": (timings["draft_n"] - timings["draft_n_accepted"]) if depth else 0,
                   "prompt_checkpoint_marker_count": len(re.findall(r"restored context checkpoint", log)),
                   "speculative_checkpoint_marker_count": len(re.findall(r"restore checkpoint\)|restoring speculative checkpoint", log))})
    return result, cycle_rows


def serial_on_routes(cold, warm):
    evidence = {}
    q1 = re.compile(r"occamy_sparse_probe dispatch calls=\d+ n_q=1 n_kv=(\d+) nkvh=2[^\n]*?revision=2")
    for phase, log in zip(STAGES, (cold, warm)):
        sampler = list(re.finditer(r"slot init_sampler:.*?tokens: text = 80000, total = 80000\b", log))
        serial = next((match for match in q1.finditer(log) if int(match[1]) >= 80000), None)
        if len(sampler) != 1 or serial is None or serial.start() <= sampler[0].end():
            raise ValueError("serial-on lacks prompt preparation followed by long Q1 decode")
        before, after = log[:serial.start()], log[serial.start():]
        prep = [row for row in original.sampled(before) if row["q"] == 4]
        if (any(row["q"] == 4 for row in original.sampled(after)) or GEOMETRY.search(after) or
                any(row["kind"] != "sparse-r2" for row in prep)):
            raise ValueError("serial-on Q4 did not stay in the prompt-prefill period")
        evidence[phase] = {"observed_q4_prefill": prep, "long_q1_decode": original.sampled(after)}
    return evidence


def routes(mode, cold, warm, full):
    if mode == "serial-on":
        geometry = [tuple(map(int, match)) for match in GEOMETRY.findall(full)]
        evidence = serial_on_routes(cold, warm)
        if len(geometry) > 1 or any((nu, threads, blocks, nkvh) != (2, 512, 1, 2)
                                    for nu, threads, _, blocks, _, nkvh in geometry):
            raise ValueError("serial-on Q4 prefill used an unexpected geometry")
        return {"warm_dispatch_samples": original.sampled(warm), "actual_q4_verifier_samples": {"cold": [], "warm": []},
                "serial_on_dispatch": evidence, "q4_geometry": dict(zip(
                    ("nu", "threads", "nsplit", "blocks_per_sm", "n_kv", "nkvh"), geometry[0])) if geometry else None}
    warm_samples = original.sampled(warm)
    warm_keys = {(row["kind"], row["q"]) for row in warm_samples}
    verifier = {}
    for phase, log in zip(STAGES, (cold, warm)):
        first_accept = original.n92.ACCEPT.search(log)
        verifier[phase] = [row for row in original.sampled(log[first_accept.end():])
                           if row["q"] == 4 and row["calls"] >= 100] if first_accept else []
    verifier_kinds = {row["kind"] for samples in verifier.values() for row in samples}
    geometry = [tuple(map(int, match)) for match in GEOMETRY.findall(full)]
    active_plans = 0
    if mode == "no-mtp":
        if ("sparse-r2", 1) not in warm_keys or ("sparse-r2", 4) in warm_keys or geometry or verifier_kinds:
            raise ValueError("long warm serial q1 sparse dispatch differs")
    elif mode == "depth3-off":
        if verifier_kinds != {"mrow"} or geometry:
            raise ValueError("long native verifier full q4 mrow dispatch differs")
    else:
        if verifier_kinds != {"sparse-r2"} or not verifier["warm"] or len(geometry) != 1:
            raise ValueError("long warm native verifier Q4 R2 dispatch or unique NU2 geometry missing")
        nu, threads, nsplit, blocks, n_kv, nkvh = geometry[0]
        if (nu, threads, blocks, nkvh) != (2, 512, 1, 2) or nsplit <= 0 or n_kv < 8192:
            raise ValueError("Q4 R2 NU2 execution geometry differs")
        plan = re.findall(
            r"occamy_sparse_plan call=1 n_q=4 row=(\d) kvh=(\d) n_kv=\d+ ok=1 invalid=0 fallback=0", full)
        if sorted(plan) != [(str(row), str(head)) for row in range(4) for head in range(2)]:
            raise ValueError("first Q4 R2 selector call did not actively span all four rows and both KV heads")
        active_plans = len(plan)
    return {"warm_dispatch_samples": warm_samples, "actual_q4_verifier_samples": verifier,
            "q4_first_call_active_plans": active_plans,
            "q4_geometry": dict(zip(("nu", "threads", "nsplit", "blocks_per_sm", "n_kv", "nkvh"), geometry[0]))
            if geometry else None}


def short_request(phase,out,log):
    record=n9.read_json(out / f"{phase}.json")
    t=record["timings"]
    ids=record["token_ids"]
    base=original.n92.corpus("2k")
    if not (record["prompt_sha256"]==base["prompt_sha256"] and
            len(ids)==t["predicted_n"]==192 and original.n92.hash_ids(ids)==record["output_sha256"] and
            record["id_slot"]==0 and record["finish_reason"]=="limit" and not record["truncated"] and
            t["prompt_n"]+t["cache_n"]==base["prompt_n"] and (t["cache_n"]==0 if phase=="cold" else 0<t["prompt_n"]<=16) and
            n9.elapsed(t["predicted_ms"]) and n9.elapsed(record["wall_s"])):
        raise ValueError("short 2K measured request incomplete or cache mismatch")
    hist=original.n92.histogram(log,3,t["draft_n_accepted"],t["draft_n"])
    return {**hist,"n_out":192,"decode_toks_s":192000/t["predicted_ms"],
            "decode_ms_per_native_cycle":t["predicted_ms"]/hist["rounds"],"request_wall_s":record["wall_s"],
            "output_sha256":record["output_sha256"],"cache_n":t["cache_n"],"evaluated_prompt_n":t["prompt_n"]}

def collect(mode, profile, out):
    maps = ready(mode, profile, out)
    start, cold, warm, full = log_phases(out)
    if not cold or not warm or original.BAD.search(full) or n18.SHIFT.search(full):
        raise ValueError("complete non-shifting GPU requests not proven")
    depth = MODES[mode]
    if (bool(n9.CACHE_MARKER.search(cold)) != bool(depth) or bool(n9.FULL_MARKER.search(full)) != bool(depth) or
            n9.CACHE_MARKER.search(start) or "MTP merge re-eval enabled" in full or "LLAMA_MTP_MERGE_REEVAL ignored" in full):
        raise ValueError("live split/cache-only mode differs")
    short=mode.startswith("smoke")
    score=None
    quality_events=[]
    if short:
        after=(out/"server-post-warm.log").read_text(errors="replace")
        before=(out/"server-pre-warm.log").read_text(errors="replace")
        if not full.startswith(after) or not after.startswith(before): raise ValueError("quality phase boundary missing")
        warm=after[len(before):]
        if n9.read_json(out/"quality-request.json")!=original.n92.quality48.quality_body(): raise ValueError("quality request changed")
        score=original.n92.quality48.score(n9.read_json(out/"quality-response.json"))
        if not score["passed"] or score!=n9.read_json(out/"quality-score.json"): raise ValueError("independent arithmetic witness failed")
        quality_events=original.n92.ACCEPT.findall(full[len(after):])
    path={"short_prompt_attention_scope":"2K below sparse4 activation threshold"} if short else routes(mode,cold,warm,full)
    phases={}
    for phase,log in zip(STAGES,(cold,warm)):
        phases[phase]=short_request(phase,out,log) if short else request(mode,phase,out,log)[0]
        begin,end=("start","cold") if phase=="cold" else ("cold","warm")
        phases[phase]["moe_dispatch"]=attribution(mode,out/f"counter-{begin}.bin",out/f"counter-{end}.bin",full,phases[phase])
    counters=[list(map(int,row)) for row in HOST.findall(full)]
    if depth and (not counters or counters[-1][0]!=len(quality_events)+sum(r["rounds"] for r in phases.values())):
        raise ValueError("final host rounds do not reconcile")
    if depth and counters[-1][1]!=sum(bool(e[2]) for e in quality_events)+sum(r["checkpoint_restore_events"] for r in phases.values()):
        raise ValueError("final host restores do not reconcile")
    if not depth and (any(rounds or restores for rounds, restores in counters) or "statistics        draft-mtp:" in full):
        raise ValueError("no-MTP arm reported speculative cycles")
    return {"mode": mode, "depth": depth, "technical_pass": True, "profile": profile,
            "fixture_sha256": original.n92.corpus("2k")["prompt_sha256"] if short else FIXTURE, "metrics": phases, "host_rounds_restores": counters[-1] if counters else None,
            "incomplete_host_counter_lines": full.count("mv1_host: rounds=") - len(counters),
            **path, "semantic_score": score,"quality_request_rounds":len(quality_events),
            "instrumented_phase_summary": phase_summary(out / "phase.tsv") if profile else None,
            "mapped_objects": {Path(path).name: digest for path, digest in maps.items()}}


def sanitize(value):
    private = {"fixture_sha256", "prompt_sha256", "output_sha256", "token_ids", "text"}
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key not in private}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def partial_record(path):
    try:
        value = n9.read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def finalize(mode, profile, out, runner, cleanup, started, finished):
    raw = out / "private"
    public = out / "sanitized"
    public.mkdir(exist_ok=True)
    try:
        _, cold, warm, full = log_phases(raw)
    except ValueError:
        cold = warm = full = ""
    cycle_rows = []
    observed = {}
    for phase, log in zip(STAGES, (cold, warm)):
        record = partial_record(raw / f"{phase}.json")
        if record and mode.startswith("smoke"):
            record["n_out"]=len(record.get("token_ids",[]))
            if phase=="warm" and (raw/"server-post-warm.log").exists():
                before=(raw/"server-pre-warm.log").read_text(errors="replace")
                after=(raw/"server-post-warm.log").read_text(errors="replace")
                if after.startswith(before): log=after[len(before):]
        complete = record.get("n_out") == (192 if phase == "cold" or mode.startswith("smoke") else 3072) and record.get("error") is None
        emitted = record.get("n_out", 0)
        emitted = emitted if type(emitted) is int and 0 <= emitted <= 3072 else 0
        phase_rows, residual = cycles(mode, phase, log, emitted, complete)
        for i, row in enumerate(phase_rows, 1 + len(cycle_rows)):
            row["arm_cycle"] = i
        cycle_rows.extend(phase_rows)
        if record:
            timing = record.get("timings")
            numeric_timing = {key: timing[key] for key in
                              ("cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
                               "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
                               "draft_n", "draft_n_accepted")
                              if isinstance(timing, dict) and type(timing.get(key)) in (int, float)}
            observed[phase] = {"n_out": emitted, "completed_request": complete,
                               **{key: record.get(key) for key in ("ttft_s", "wall_s")
                                  if type(record.get(key)) in (int, float)}, "timings": numeric_timing,
                               "unassigned_serial_outputs": residual}
    with (private_cycles := raw / "cycles.jsonl").open("x") as file, (public / "cycles.jsonl").open("x") as safe:
        for row in cycle_rows:
            line = json.dumps(row, separators=(",", ":")) + "\n"
            file.write(line)
            safe.write(line)
    status = {"runner_exit": runner, "cleanup_exit": cleanup, "bounded_wall_s": finished - started}
    status["technical_pass"] = runner == cleanup == 0 and (raw / "result.json").is_file()
    # Allowlist only, including on failures. Paths, raw logs, token IDs and text never enter the export.
    report = n9.read_json(raw / "result.json") if status["technical_pass"] else None
    env, args = settings(mode, profile)
    runtime_maps = None
    if (raw / "runtime-library-hashes.txt").is_file():
        try:
            runtime_maps = {Path(path).name: digest for path, digest in mapped(raw).items()}
        except ValueError:
            pass
    n9.write_json(public / "receipt.json", {
        "format_version": 1, "mode": mode, "cap_seconds": 190 if mode.startswith("smoke") else 430, "model_sha256": n9.MODEL_SHA,
        "image_sha256": n9.IMAGE.removeprefix("sha256:"),
        "gpu_requested": "Quadro M6000 24GB", "fleet_admitted_cap_w": 250 if (raw / "admission.txt").is_file()
        and n9.IDLE_MARKER.search((raw / "admission.txt").read_text()) else None,
        "native_q4_cuda_sha256": Q4_CUDA, "runtime_maps": runtime_maps,
        "configured_argv": args, "configured_environment": env[1::2],
        "provenance_sha256": n9.hash_file(HERE / "FROZEN.sha256"),
        "profile_instrumented": profile, "status": status, "observed_requests": observed,
        "speculative_trace_marker_count": len(original.n92.ACCEPT.findall(full)),
        "cycle_rows_sha256": n9.hash_file(private_cycles),
        "validated_report": sanitize(report),
    })


def compare(paths):
    results={};ids={}
    for path in paths:
        run=n9.read_json(path/"sanitized/receipt.json")
        if not run["status"]["technical_pass"] or run["profile_instrumented"]: raise ValueError("incomplete or profiled arm")
        mode=run["mode"];raw=path/"private"
        if mode in results: raise ValueError("duplicate mode")
        r=collect(mode,False,raw)
        if json.loads(json.dumps(r))!=n9.read_json(raw/"result.json"): raise ValueError("stored result differs")
        results[mode]=r;ids[mode]={p:n9.read_json(raw/f"{p}.json")["token_ids"] for p in STAGES}
    comparisons={}
    for key,base,trial in (("attention_only","depth3-off","depth3-attn"),("moe_over_attention","depth3-attn","depth3-combined"),
            ("combined_over_off","depth3-off","depth3-combined"),("strict_native_over_serial","serial-on","depth3-combined"),
            ("short_combined_over_off","smoke-off","smoke-on")):
        if base not in results or trial not in results: continue
        if key=="strict_native_over_serial" and settings(base,False)[0]!=settings(trial,False)[0]: raise ValueError("strict target environment differs")
        comparisons[key]={}
        for p in STAGES:
            a=results[base]["metrics"][p];b=results[trial]["metrics"][p]
            comparisons[key][p]={"decode_ratio":b["decode_toks_s"]/a["decode_toks_s"],
                "base_minus_trial_wall_s":a["request_wall_s"]-b["request_wall_s"],
                "base_minus_trial_ms_per_native_cycle":a["decode_ms_per_native_cycle"]-b["decode_ms_per_native_cycle"] if results[base]["depth"] else None,
                "first_different_id_1based":n9.first_mismatch(ids[base][p],ids[trial][p])}
    ratio=comparisons.get("strict_native_over_serial",{}).get("warm",{}).get("decode_ratio")
    return {"arms":{k:sanitize(r) for k,r in results.items()},"comparisons":comparisons,"strict_warm_2x":ratio>=2 if ratio is not None else None}

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    for name in ("env", "args", "ready", "verify", "finalize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("mode", choices=MODES)
        cmd.add_argument("profile", type=int, choices=(0, 1))
        if name in ("ready", "verify", "finalize"):
            cmd.add_argument("out", type=Path)
        if name == "finalize":
            for key in ("runner", "cleanup", "started", "finished"):
                cmd.add_argument(key, type=int)
    cmd = sub.add_parser("compare")
    cmd.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        qualify()
        check_moe()
        print("Pinned 80K, archived N100 r1 and passed Q4 NU2 operator revalidated")
    elif args.action in ("env", "args"):
        print("\n".join(settings(args.mode, args.profile)[args.action == "args"]))
    elif args.action == "ready":
        ready(args.mode, args.profile, args.out)
    elif args.action == "verify":
        n9.write_json(args.out / "result.json", collect(args.mode, bool(args.profile), args.out))
    elif args.action == "finalize":
        finalize(args.mode, bool(args.profile), args.out, args.runner, args.cleanup, args.started, args.finished)
    else:
        print(json.dumps(compare(args.paths), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, TypeError) as exc:
        sys.exit(f"Native width: {exc}")
