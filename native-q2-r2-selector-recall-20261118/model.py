"""Run the existing independent 80K midpoint fixture and validate selector use."""
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
BASE = C / "last-ms-r2-selector-model"
ARCHIVE = C / "night-gdn-q8-long/receipts/recall-20261118T041839Z-2992636-18432"
spec = importlib.util.spec_from_file_location("selector_model", BASE / "model.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
n88, n9, read, need = probe.prior.n88, probe.n9, probe.read, probe.need

def check():
    probe.check()
    n88.check_reference()
    ref = n88.inspect_recall(ARCHIVE)
    need(ref["technical_pass"] and ref["exact_seven_ids_to_N50"] and ref["candidate_score"]["recall_pass"],
         "reference 80K midpoint has no exact scored evidence")
    need("n_q=2 n_kv=80128 nkvh=2 " in (ARCHIVE / "server.log").read_text(),
         "archived scored fixture did not engage eligible Q2")

def verify(out):
    original = n88.startup
    n88.startup = lambda stage, path: probe.prior.ready("native", path)
    try:
        observed = n88.inspect_recall(out)
    finally:
        n88.startup = original
    tokens = read(out / "private/response.json")["tokens"]
    reference = read(ARCHIVE / "private/response.json")["tokens"]
    score = observed["candidate_score"]
    need(score["recall_pass"] and observed["exact_seven_ids_to_N50"] and tokens == reference and len(tokens) == 7,
         "exact scored fact, EOS or seven output IDs differ from reference")
    sampled = [read(out / "witness" / f"{label}.json") for label in ("start", "cold")]
    child = int((out / "witness/child.pid").read_text())
    names = ("row0", "row1", "combined", "skip", "q2_sparse")
    need(all(s["child"] == child and all(type(s[k]) is int and s[k] >= 0 for k in names) for s in sampled),
         "invalid witnessed owner, counter type or value")
    deltas = {k: sampled[1][k] - sampled[0][k] for k in names}
    timing = observed["request"]["timings"]
    offered, accepted = timing["draft_n"], timing["draft_n_accepted"]
    need(offered > 0 and 0 < accepted <= offered, "scored midpoint did not exercise native verification")
    need(sampled[0]["q2_sparse"] == 0 and deltas["q2_sparse"] == 10 * offered
         and deltas["row0"] == deltas["row1"] == 0 and deltas["combined"] == deltas["skip"] == 40 * offered,
         "physical MoE or sparse Q2 count differs from offered verifier rounds")
    start = (out / "server-start.log").read_text()
    final = (out / "server.log").read_text()
    phase = final[len(start):]
    rows = [tuple(map(int, r)) for r in probe.DISPATCH.findall(phase)]
    need(probe.TOP16.findall(final) == probe.TOP16.findall(phase) == ["80128"] and rows
         and len(rows) == len(probe.ANY_Q2.findall(phase)) and all(row[1:] == (80128, 2, 2) for row in rows),
         "new selector or eligible 80,128 Q2 sparse route missing from the scored request")
    return {
        "technical_pass": True, "expected_fact_match": score["recall_pass"],
        "eos_terminated": score["eos_terminated"], "exact_output_ids_to_n50_and_n88": len(tokens),
        "source_fact_position": 40000, "prompt_tokens_evaluated": timing["prompt_n"],
        "cached_prompt_tokens": timing["cache_n"], "requested_wall_s": observed["request"]["wall_s"],
        "offered_drafts": offered, "accepted_drafts": accepted, "physical_request_counter_deltas": deltas,
        "sparse_q2_dispatch_samples": [{"calls": r[0], "width": r[1], "kv_heads": r[2], "revision": r[3]} for r in rows],
        "first_fast_selector_width": 80128, "candidate_software_sha256": probe.DSO,
        "experts": 8, "gpu_layers_offloaded": 42,
        "scope": "One independent known midpoint fact. The first dispatch witnesses the new selector. The Q2 counter covers aggregate sparse selection.",
    }

def finish(out, rc, clean, elapsed):
    safe = out / "sanitized"
    safe.mkdir()
    result = same = None
    if rc == clean == 0:
        result = read(out / "result.json")
    elif (out / "run.log").is_file():
        for line in (out / "run.log").read_text().splitlines():
            if re.fullmatch(r"Selector recall: [A-Za-z0-9/ _.-]{1,140}", line):
                same = line.split(": ", 1)[1]
    n9.write_json(safe / "model-receipt.json", {
        "mode": "recall", "runner_exit": rc, "cleanup_exit": clean, "bounded_wall_s": elapsed,
        "validation": result, "refusal": same,
    })

if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    try:
        if cmd == "check":
            check()
        elif cmd in ("env", "args"):
            need(args == ["recall"], "unexpected stage")
            print("\n".join(probe.settings("native")[cmd == "args"]))
        elif cmd == "ready":
            probe.prior.ready("native", Path(args[1]))
        elif cmd == "request":
            n88.n7.run(int(args[0]), Path(args[1]), n88.n48.N50)
        elif cmd == "verify":
            out = Path(args[1])
            n9.write_json(out / "result.json", verify(out))
        elif cmd == "finalize":
            finish(Path(args[1]), *map(int, args[2:]))
        else:
            raise ValueError("unknown typed recall command")
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit("Selector recall: " + str(exc)) from None
