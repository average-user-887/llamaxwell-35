"""Preserve the typed Q2+Q8 model checks while binding the exact selector candidate."""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
BASE = C / "resume-q2-moe-native-r1"
OPERATOR = C / "last-ms-r2-selector"
DSO = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"
OLD_DSO = "19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00"
REFERENCE = {
    "serial": BASE / "receipts/serial-20261118T142928Z-362150-30056",
    "native": BASE / "receipts/native-20261118T142401Z-347777-15483",
}
OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16"
spec = importlib.util.spec_from_file_location("typed_q2", BASE / "model.py")
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

def settings(stage):
    env, args = original_settings(stage)
    need(not any(item.startswith(OPTION) for item in env), "selector flag already occupied")
    return [*env, "-e", OPTION + "=1", "-e", OPTION + "_CHECK=0"], args

prior.DSO = DSO
prior.SUP = HERE / "supervise"
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
             f"{phase}: exact token IDs, full text or draft/accept counts differ from prior same-mode 19607daf")
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
    result["outputs_vs_previous_same_mode_19607daf"] = exact_reference(stage, out)
    result["selector_execution_witness"] = physical_witness(stage, out, result["metrics"])
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
            "native_minus_prior_19607daf_decode_toks_s": current_n - historic_n,
            "serial_minus_prior_19607daf_decode_toks_s": current_s - historic_s,
            "native_saved_ms_per_round_vs_19607daf": on["predicted_ms"] / on["draft_n"] - un["predicted_ms"] / un["draft_n"],
            "same_build_native_vs_serial_first_id_difference": n9.first_mismatch(current[0]["token_ids"], current[1]["token_ids"]),
            "same_build_native_vs_serial_full_text_equal": current[0]["text"] == current[1]["text"],
            "native_selector_witness": b["selector_execution_witness"]["phases"][phase],
            "serial_selector_witness": a["selector_execution_witness"]["phases"][phase],
        }
    return {"candidate_cuda_sha256": DSO, "prior_cuda_sha256": OLD_DSO,
            "technical_pass": True, "semantic_score": None,
            "exact_output_ids_same_mode_each": 3264,
            "warm_same_build_25_percent_gate": rows["warm"]["same_build_native_over_serial_ratio"] >= 1.25,
            "measurements": rows}

def check():
    need(n9.hash_file(OPERATOR / "libggml-cuda.so") == DSO, "candidate identity")
    op = read(OPERATOR / "receipts/20261118T151601Z-483850/result.json")
    need(op["physical_gate_pass"] and op["candidate_sha256"] == DSO and op["control_sha256"] == OLD_DSO,
         "standalone physical gate absent or library differs")
    for stage, path in REFERENCE.items():
        value = read(path / "result.json")
        need(value["technical_pass"] and value["mode"] == stage
             and value["actual_server_software_sha256"]["libggml-cuda.so"] == OLD_DSO, "prior model identity")
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
