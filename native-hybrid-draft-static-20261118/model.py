#!/usr/bin/env python3
"""Bind the two N40 draft libraries to the exact corrected finish33 model checks."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
C = HERE.parent
R1 = C / "finish33-sampler-r1"
binding = json.loads((HERE / "binding.json").read_text())
baseline = C / binding["native_baseline"]
original = binding["original_libraries"]
alternate = binding["alternate_libraries"]
libs = C / "night-cache-only/libs"

spec = importlib.util.spec_from_file_location("hybrid_finish33_r1", R1 / "validate.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
core = validator.model
q2 = core.prior
need = core.need
MERGE = "MTP merge re-eval enabled"
NARROW = re.compile(
    r"^(?:\d+\.\d{2}\.\d{3}\.\d{3} I )?mtp_last_graph pos=80002 rows=3 outputs=1 "
    r"type=5 kw=3 vw=3 qproj=1 q=1 nkv=80128 mask=1 ffn=1 ordered=1$", re.M)
INPUT_FIELDS = ("prompt_sha256", "prompt_n", "n_predict", "seed", "temperature", "cache_prompt",
                "ignore_eos", "finish_reason", "truncated", "error")


def settings(stage):
    need(stage == "native", "only the native depth-one hybrid is admitted")
    env, args = core.settings(stage)
    need(env.count("LLAMA_MTP_MERGE_REEVAL=0") == 1 and
         not any(item.startswith("LLAMA_MTP_LAST_QUERY") for item in env), "original split profile differs")
    need((baseline / "environment.args").read_text().splitlines() == env and
         (baseline / "server.args").read_text().splitlines() == args, "archived current profile differs")
    env = ["LLAMA_MTP_MERGE_REEVAL=1" if item == "LLAMA_MTP_MERGE_REEVAL=0" else item for item in env]
    return [*env, "-e", "LLAMA_MTP_LAST_QUERY=1"], args


def confirm_merge(start):
    need(start.count(MERGE) == 1 and "LLAMA_MTP_MERGE_REEVAL ignored" not in start,
         "hybrid did not enable merged draft cache catch-up")


@contextmanager
def candidate_bindings():
    """Adapt only the expected library hashes and the split-only route marker."""
    prior_settings = q2.settings
    prior_maps = q2.base.maps
    prior_dispatch = q2.n88.dispatch

    def expected_maps(path):
        observed = prior_maps(path)
        if path == q2.REF / "runtime-library-hashes.txt":
            for name, digest in alternate.items():
                key = str(libs / name)
                need(observed[key] == original[name], "original mapped draft library differs")
                observed[key] = digest
        return observed

    def hybrid_dispatch(stage, full, phase):
        confirm_merge(full)
        return prior_dispatch(stage, full.replace(MERGE, "MTP merged draft route witnessed", 1), phase)

    q2.settings = settings
    q2.base.maps = expected_maps
    q2.n88.dispatch = hybrid_dispatch
    try:
        yield
    finally:
        q2.settings = prior_settings
        q2.base.maps = prior_maps
        q2.n88.dispatch = prior_dispatch


def ready(stage, out):
    with candidate_bindings():
        start, observed = q2.ready(stage, out)
    confirm_merge(start)
    return observed


def graph_witness(out):
    start, cold, full = [(out / name).read_text() for name in
                         ("server-start.log", "server-pre-warm.log", "server.log")]
    need("mtp_last_graph" not in start and cold.count("mtp_last_graph") == full.count("mtp_last_graph") == 1
         and len(NARROW.findall(cold)) == len(NARROW.findall(full)) == 1,
         "exact first 80K one-graph K/V and narrow query witness absent")
    return {"cold_first_position": 80002, "draft_rows": 3, "k_write_rows": 3, "v_write_rows": 3,
            "query_rows": 1, "ffn_rows": 1, "key_width": 80128, "ordered_cache_writes": True,
            "one_first_dispatch_marker": True}


def exact_parity(out):
    parity = {}
    for phase, expected in (("cold", 192), ("warm", 3072)):
        a, b = [core.read(root / f"{phase}.json") for root in (baseline, out)]
        need(all(a[key] == b[key] for key in INPUT_FIELDS), f"{phase}: request fixture differs")
        same_totals = all(a["timings"][key] == b["timings"][key] for key in (
            "draft_n", "draft_n_accepted", "cache_n", "prompt_n"))
        need(a["token_ids"] == b["token_ids"] and len(a["token_ids"]) == expected and a["text"] == b["text"]
             and same_totals, f"{phase}: exact token IDs, full text or draft/accept counts differ from clean N1 split repeat")
        parity[phase] = {"all_ids_equal": True, "compared_ids": expected, "full_text_equal": True,
                         "same_draft_acceptance_and_cache_totals": True}
    return parity


def collect(stage, out):
    exact = exact_parity(out)
    graph = graph_witness(out)
    with candidate_bindings():
        result = core.collect(stage, out)
    result["draft_variant"] = "merged catch-up with N40 single final query"
    result["draft_graph_witness"] = graph
    result["exact_output_vs_clean_current_split"] = exact
    return result


def result_comparison(out):
    need(out.parent == HERE / "receipts", "candidate must be an owned terminal receipt")
    validator.sha_check(out, "observations.sha256")
    statuses = (out / "status.txt").read_text().splitlines()
    need({"runner_exit=0", "cleanup_exit=0"} <= set(statuses), "candidate terminal run or cleanup failed")
    result = collect("native", out)
    need(result == core.read(out / "result.json"), "candidate terminal result no longer reproduces")
    reference = validator.verify("native", baseline)["validated"]
    phases = {}
    for phase in ("cold", "warm"):
        old, new = (value["metrics"][phase] for value in (reference, result))
        phases[phase] = {"control_decode_tok_s": old["decode_toks_s"], "hybrid_decode_tok_s": new["decode_toks_s"],
                         "control_wall_s": old["request_wall_s"], "hybrid_wall_s": new["request_wall_s"],
                         "saved_ms_per_offer": old["decode_ms_per_round"] - new["decode_ms_per_round"],
                         "observed_warm_100_gate": phase == "warm" and new["decode_toks_s"] >= 100,
                         "exact": result["exact_output_vs_clean_current_split"][phase]}
    return {"scope": "clean archived split repeat versus one separately launched hybrid",
            "hardware_or_other_routes_changed": False, "semantic_score": None,
            "observed_candidate_software_sha256": result["actual_server_software_sha256"], "phases": phases}


def check():
    reference = validator.verify("native", baseline)["validated"]
    need(reference["actual_server_software_sha256"] == binding["baseline_mapped_software"],
         "unmodified current repeat no longer matches staged provenance")
    need(reference["metrics"]["warm"]["draft_n"] == 1642 and reference["metrics"]["warm"]["n_out"] == 3072,
         "clean current repeat fixture changed")
    settings("native")


def main():
    mode, *args = sys.argv[1:]
    if mode == "check" and not args:
        check()
    elif mode in ("env", "args"):
        print("\n".join(settings(args[0])[mode == "args"]))
    elif mode == "ready":
        ready(args[0], Path(args[1]))
    elif mode == "verify":
        core.n9.write_json(Path(args[1]) / "result.json", collect(args[0], Path(args[1])))
    elif mode == "finalize":
        q2.finish(args[0], Path(args[1]), *map(int, args[2:]))
    elif mode == "compare":
        print(json.dumps(result_comparison(Path(args[0]).resolve()), indent=2, allow_nan=False))
    else:
        raise ValueError("unknown hybrid draft command")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Q2 model: {exc}") from None
