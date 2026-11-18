"""Preserve the typed Q2+Q8 model checks while binding the exact selector candidate."""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
BASE = C / "last-ms-r2-selector-model"
OPERATOR = C / "finish33-r2-stride"
DSO = "369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871"
PREVIOUS_DSO = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"
REFERENCE = {
    "serial": BASE / "receipts/@OWNER_PRIOR_SELECTOR_SERIAL_RECEIPT@",
    "native": BASE / "receipts/@OWNER_PRIOR_SELECTOR_NATIVE_RECEIPT@",
}
OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16"
spec = importlib.util.spec_from_file_location("typed_q2", HERE / "q2_model.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read = prior.read
need = prior.need
n9 = prior.n9
original_settings = prior.settings

DISPATCH = re.compile(r"^ggml_cuda: occamy_sparse_probe dispatch calls=(\d+) n_q=2 n_kv=(\d+) nkvh=(\d+) "
                      r"nsplit=12 blocks_per_sm=1 approximate=1 keep_old_blocks=16 recent=4096 revision=(\d+)$", re.M)
ANY_Q2 = re.compile(r"^ggml_cuda: occamy_sparse_probe dispatch calls=\d+ n_q=2\b.*$", re.M)
TOP16 = re.compile(r"^q2_r2_top16_dispatch n_q=2 n_kv=(\d+) nkvh=2 plans=4$", re.M)
WIDTHS = set(range(80128, 83201, 256))

SNAPSHOT_OPTION = "LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE"
SNAPSHOT = re.compile(
    r"^sampler_snapshot: elided=(\d+) elided_rejected=(\d+) kept=(\d+) restored=(\d+)$",
    re.M,
)


def snapshot_witness(stage, out, metrics):
    names = ("server-start.log", "server-pre-warm.log", "server.log")
    observed = []
    for name in names:
        log = (out / name).read_text()
        if name == "server-start.log" and stage == "native":
            need("the context supports bounded partial sequence removal" in log,
                 "native target rollback capability is not bounded partial")
        restores = re.findall(r"^mv1_host: rounds=\d+ restores=(\d+)\b", log, re.M)
        need(all(int(count) == 0 for count in restores), "legacy full-checkpoint restore occurred")
        if name == "server.log" and stage == "native":
            need(restores, "legacy restore counter missing")
        rows = [tuple(map(int, row)) for row in SNAPSHOT.findall(log)]
        need(len(rows) == log.count("sampler_snapshot:"), "malformed snapshot witness")
        observed.append(rows)
    startup, cold, final = observed
    need(not startup and len(cold) == 1 and len(final) == 2 and final[:1] == cold,
         "snapshot witness does not match the two completion boundaries")
    result = {}
    previous = (0, 0, 0, 0)
    for phase, current in zip(("cold", "warm"), final):
        delta = tuple(a - b for a, b in zip(current, previous))
        offered = metrics[phase]["draft_n"] or 0
        accepted = metrics[phase]["draft_n_accepted"] or 0
        expected = (offered, offered - accepted, 0, 0) if stage == "native" else (0, 0, 0, 0)
        need(delta == expected, f"{phase}: sampler snapshots, rejected drafts or restores differ")
        result[phase] = dict(zip(("elided", "elided_rejected", "kept", "restored"), delta))
        previous = current
    return result

PROBE_OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4"
STRIDE_OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE"
PROBE4 = re.compile(r"^q2_r2_probe4_dispatch width=(\d+) plans=4 blocks=(\d+) tiles=(\d+)$", re.M)
STRIDE = re.compile(
    r"^q2_r2_stride_dispatch width=(\d+) k_token_bytes=(\d+) v_token_bytes=(\d+) splits=12 nu=4 threads=512 grid=24$",
    re.M,
)


def stride_witness(stage, out, metrics, selector_witness):
    logs = [(out / name).read_text() for name in ("server-start.log", "server-pre-warm.log", "server.log")]
    markers = []
    for log in logs:
        probe = [tuple(map(int, row)) for row in PROBE4.findall(log)]
        stride = [tuple(map(int, row)) for row in STRIDE.findall(log)]
        need(len(probe) == log.count("q2_r2_probe4_dispatch"),
             "missing or invalid tiled probe dispatch geometry")
        need(len(stride) == log.count("q2_r2_stride_dispatch"),
             "missing or invalid fixed-stride dispatch geometry")
        markers.append((probe, stride))
    startup, cold, final = markers
    need(startup == ([], []) and cold == final, "stride markers outside first measured request")
    pitch = None
    if stage == "native":
        need(len(cold[0]) == len(cold[1]) == 1, "tiled probe or fixed-stride path did not dispatch")
        (probe_width, blocks, tiles), = cold[0]
        (stride_width, k_pitch, v_pitch), = cold[1]
        width = selector_witness["branch_first_dispatch_width"]
        need(probe_width == stride_width == width and blocks == width // 256 and tiles == (blocks + 3) // 4,
             "first selector, tiled probe and stride widths differ")
        need(k_pitch == v_pitch and k_pitch in (1024, 2048), "stride pitch outside physical coverage")
        pitch = k_pitch
    else:
        need(cold == ([], []), "serial executed a two-row tiled or stride specialization")

    snapshots = [read(out / "witness" / f"{phase}.json") for phase in ("start", "cold", "warm")]
    counts = [snapshot.get("q2_stride") for snapshot in snapshots]
    need(all(type(count) is int and count >= 0 for count in counts) and counts[0] == 0,
         "invalid initial or typed stride count")
    phases = {}
    for phase, start, end in zip(("cold", "warm"), counts, counts[1:]):
        expected = 10 * (metrics[phase]["draft_n"] or 0)
        need(end - start == expected, f"{phase}: fixed-stride engagement differs from ten per offered round")
        phases[phase] = {"actual_stride_dispatches": end - start, "expected_stride_dispatches": expected}
    return {"first_physical_kv_pitch_bytes": pitch, "phases": phases}


def settings(stage):
    env, args = original_settings(stage)
    need(not any(item.startswith(OPTION) for item in env), "selector flag already occupied")
    return [*env, "-e", OPTION + "=1", "-e", OPTION + "_CHECK=0", "-e", PROBE_OPTION + "=1", "-e", STRIDE_OPTION + "=1",
            "-e", SNAPSHOT_OPTION + "=1"], args

prior.DSO = DSO
prior.SUP = HERE / "supervise"
SERVER_IMPL_SHA = read(HERE / "build.json")["candidate_server_impl_sha256"]
prior.SERVER_IMPL_SHA = SERVER_IMPL_SHA
prior.settings = settings

def exact_reference(stage, out):
    report = {}
    for phase, count in (("cold", 192), ("warm", 3072)):
        a = read(out / f"{phase}.json")
        b = read(REFERENCE[stage] / f"{phase}.json")
        ids_equal = a["token_ids"] == b["token_ids"]
        text_equal = a["text"] == b["text"]
        draft_fields = ("draft_n", "draft_n_accepted")
        repeated_totals = all(a["timings"].get(k) == b["timings"].get(k) for k in draft_fields)
        need(len(a["token_ids"]) == count and ids_equal and text_equal and repeated_totals,
             f"{phase}: exact token IDs, full text or draft/accept counts differ from prior same-mode selector")
        report[phase] = {"ids_compared": count, "all_token_ids_equal": ids_equal, "full_text_equal": text_equal,
                         "same_draft_and_acceptance_totals": repeated_totals}
    return report

def physical_witness(stage, out, metrics):
    snapshots = [read(out / "witness" / f"{name}.json") for name in ("start", "cold", "warm")]
    counts = [sample["q2_sparse"] for sample in snapshots]
    need(all(type(n) is int and n >= 0 for n in counts) and counts[0] == 0 and counts == sorted(counts),
         "nonzero initial or backwards sparse Q2 aggregate")
    startup = (out / "server-start.log").read_text()
    prewarm = (out / "server-pre-warm.log").read_text()
    full = (out / "server.log").read_text()
    need(not ANY_Q2.search(startup) and "q2_r2_top16_dispatch" not in startup, "Q2 selector ran before measured requests")
    parsed = [tuple(map(int, row)) for row in DISPATCH.findall(full)]
    cold = [tuple(map(int, row)) for row in DISPATCH.findall(prewarm)]
    need(len(parsed) == len(ANY_Q2.findall(full)) and len(cold) == len(ANY_Q2.findall(prewarm)),
         "unexpected Q2 sparse geometry, route, or probe revision")
    need(parsed[:len(cold)] == cold, "cold Q2 dispatch records missing from final log")
    need(all(k in WIDTHS and heads == 2 and revision == 2 for _, k, heads, revision in parsed),
         "sampled Q2 sparse width or runtime profile outside the approved selector")
    need([r[0] for r in parsed] == sorted(set(r[0] for r in parsed)), "unexpected Q2 dispatch count order")
    top16 = list(map(int, TOP16.findall(full)))
    if stage == "native":
        need(len(top16) == 1 and top16[0] in WIDTHS and TOP16.findall(prewarm) == [str(top16[0])],
             "expected cold first-call fast-selector launch missing")
        need(cold and cold[0][:2] == (1, top16[0]) and parsed[len(cold):], "Q2 cold or warm dispatch missing")
    else:
        need(not parsed and not top16 and "q2_r2_top16_dispatch" not in full and counts == [0, 0, 0],
             "serial path executed two-row sparse attention")
    by_phase = {}
    for phase, begin, end, sample in zip(("cold", "warm"), counts, counts[1:], (cold, parsed[len(cold):])):
        delta = end - begin
        offered = int(metrics[phase].get("draft_n") or 0)
        expected = 10 * offered
        need(delta == expected, f"{phase}: existing Q2 sparse aggregate differs from ten per offered verifier round")
        need(all(begin < i <= end for i, _, _, _ in sample), "sampled Q2 dispatch outside its request boundary")
        by_phase[phase] = {"aggregate_q2_sparse_calls": delta, "offered_drafts": offered,
                           "sampled_dispatch_count": len(sample), "sampled_key_widths": sorted({k for _, k, _, _ in sample})}
    return {
        "effective_opt_in": True, "branch_first_dispatch_width": top16[0] if top16 else None,
        "phases": by_phase,
        "scope": "Existing phase-bounded aggregate counts for sparse Q2 and sampled widths. The first launch marker directly witnesses the new branch. There is no per-call fast-selector counter or phase timer.",
    }

def collect(stage, out):
    result = prior.collect(stage, out)
    result["outputs_vs_previous_same_mode_selector"] = exact_reference(stage, out)
    result["selector_execution_witness"] = physical_witness(stage, out, result["metrics"])
    result["sampler_snapshot_witness"] = snapshot_witness(stage, out, result["metrics"])
    result["stride_execution_witness"] = stride_witness(stage, out, result["metrics"], result["selector_execution_witness"])
    return result

def finished(out):
    value = read(out / "result.json")
    need({"runner_exit=0", "cleanup_exit=0"} <= set((out / "status.txt").read_text().splitlines()),
         "new arm did not finish cleanly")
    need(collect(value["mode"], out) == value, "new receipt no longer reproduces")
    return value

def compare(serial, native):
    a, b = finished(serial), finished(native)
    need(a["mode"] == "serial" and b["mode"] == "native", "misordered arms")
    rows = {}
    for phase in ("cold", "warm"):
        current = [read(path / f"{phase}.json") for path in (serial, native)]
        old = [read(REFERENCE[stage] / f"{phase}.json") for stage in ("serial", "native")]
        us, un = (item["timings"] for item in current)
        os_, on = (item["timings"] for item in old)
        current_s, current_n, historic_s, historic_n = [
            len(item["token_ids"]) * 1000 / item["timings"]["predicted_ms"] for item in (*current, *old)
        ]
        rows[phase] = {
            "serial": a["metrics"][phase], "native": b["metrics"][phase],
            "serial_decode_toks_s": current_s, "native_decode_toks_s": current_n,
            "same_build_native_over_serial_ratio": current_n / current_s,
            "same_build_native_over_serial_pct": (current_n / current_s - 1) * 100,
            "native_minus_prior_selector_decode_toks_s": current_n - historic_n,
            "serial_minus_prior_selector_decode_toks_s": current_s - historic_s,
            "native_saved_ms_per_round_vs_prior_selector": on["predicted_ms"] / on["draft_n"] - un["predicted_ms"] / un["draft_n"],
            "same_build_native_vs_serial_first_id_difference": n9.first_mismatch(current[0]["token_ids"], current[1]["token_ids"]),
            "same_build_native_vs_serial_full_text_equal": current[0]["text"] == current[1]["text"],
            "native_selector_witness": b["selector_execution_witness"]["phases"][phase],
            "serial_selector_witness": a["selector_execution_witness"]["phases"][phase],
            "native_stride_dispatches": b["stride_execution_witness"]["phases"][phase],
            "native_first_kv_pitch_bytes": b["stride_execution_witness"]["first_physical_kv_pitch_bytes"],
            "serial_stride_dispatches": a["stride_execution_witness"]["phases"][phase],
            "native_sampler_snapshots": b["sampler_snapshot_witness"][phase],
            "serial_sampler_snapshots": a["sampler_snapshot_witness"][phase],
        }
    return {"candidate_cuda_sha256": DSO, "prior_cuda_sha256": PREVIOUS_DSO, "candidate_server_impl_sha256": SERVER_IMPL_SHA,
            "technical_pass": True, "semantic_score": None,
            "exact_output_ids_same_mode_each": 3264,
            "warm_same_build_25_percent_gate": rows["warm"]["same_build_native_over_serial_ratio"] >= 1.25,
            "measurements": rows}

def check():
    need(n9.hash_file(HERE / "libllama-server-impl.so") == SERVER_IMPL_SHA, "server implementation candidate identity")
    need(n9.hash_file(OPERATOR / "libggml-cuda.so") == DSO, "candidate identity")
    op = read(OPERATOR / "receipts/20261118T171658Z-790549/result.json")
    need(op["physical_gate_pass"] is False and op["candidate_sha256"] == DSO and
         op["control_sha256"] == PREVIOUS_DSO and op["saved_us_per_mirror"] == {"a": 27.18, "b": 26.72},
         "standalone partial operator receipt or library differs")
    replay = op["independent_candidate_replay"]
    need(set(replay) == {"a-candidate", "b-candidate"} and
         all(row["bitwise_mismatches"] == 0 and row["padded_pitch"] == 2048 and
             row["dense_replay_pitch"] == 1024 for row in replay.values()),
         "standalone parity for both physical pitches absent")
    for stage, path in REFERENCE.items():
        value = read(path / "result.json")
        need(value["technical_pass"] and value["mode"] == stage
             and value["actual_server_software_sha256"]["libggml-cuda.so"] == PREVIOUS_DSO
             and value["actual_server_software_sha256"]["libllama-server-impl.so"] ==
             read(HERE / "build.json")["original_relink_sha256"], "prior model identity")
        exact_reference(stage, path)
    native, native_args = settings("native")
    serial, serial_args = settings("serial")
    need(native == serial, "serial/native flags differ")
    need(native_args != serial_args, "model comparison has identical speculation arguments")

if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    try:
        if cmd == "check":
            check()
        elif cmd in ("env", "args"):
            print("\n".join(settings(args[0])[cmd == "args"]))
        elif cmd == "ready":
            prior.ready(args[0], Path(args[1]))
        elif cmd == "verify":
            n9.write_json(Path(args[1]) / "result.json", collect(args[0], Path(args[1])))
        elif cmd == "compare":
            print(json.dumps(compare(*map(Path, args)), indent=2))
        elif cmd == "finalize":
            prior.finish(args[0], Path(args[1]), *map(int, args[2:]))
        else:
            raise ValueError("unknown mode")
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit("Q2 model: " + str(exc)) from None
