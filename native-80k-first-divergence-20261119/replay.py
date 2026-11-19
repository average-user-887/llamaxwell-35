"""Verify the anonymized packet and independently derive the pairwise score reversal."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
read = lambda file: json.loads((ROOT / file).read_text())
pair, runtime, correction = map(read, ("divergence.json", "runtime.json", "serial-revalidation.json"))
rows = [json.loads(line) for line in (ROOT / "rows.jsonl").read_text().splitlines()]
assert len(rows) == 46
assert runtime["serial"]["completed_epoch"] <= runtime["native"]["start_epoch"]
assert runtime["serial"]["mapped_sha256"] == runtime["native"]["mapped_sha256"]
assert correction["original_refusal_reproduced"] == "malformed snapshot witness"
assert correction["original_runner_exit"] == 1 and correction["original_cleanup_exit"] == 0
assert correction["timestamp_only_revalidation"] == "pass" and correction["score_lines"] == 22
assert runtime["native"]["runner_exit"] == runtime["native"]["cleanup_exit"] == 0
assert pair["recurrent_state_bytes_compared"] is None
for phase in ("cold", "warm"):
    diff = {}
    p = pair["phases"][phase]
    assert p["shared_output_prefix_length"] == 169 and p["exact_per_arm_archive_parity"]
    for mode in ("native", "serial"):
        rr = [r for r in rows if r["mode"] == mode and r["phase"] == phase]
        assert len(rr) == (12 if mode == "native" else 11)
        for row in rr:
            assert len(row["logp_descending"]) == len(row["labels"]) == 8
            assert row["logp_descending"] == sorted(row["logp_descending"], reverse=True)
        decision, = [r for r in rr if r["committed_ordinal"] == 170]
        reference = p[mode]
        assert decision["row"] == reference["row_zero_based"] == (mode == "native")
        assert decision["n_pre"] == reference["already_emitted"] == (168 if mode == "native" else 169)
        assert decision["chosen_label"] == decision["labels"][0] == mode+"_choice" and decision["chosen_rank"] == 1
        assert reference["emitted_equals_raw_top1"]
        assert [x["log_probability"] for x in reference["top_eight"]] == decision["logp_descending"]
        assert [x["label"] for x in reference["top_eight"]] == decision["labels"]
        diff[mode] = round(decision["logp_descending"][0]-decision["logp_descending"][1],5)
        assert diff[mode] == reference["own_minus_cross"]
        assert runtime[mode]["layer_offload"] == [42,42] and runtime[mode]["active_experts"] == 8
        assert runtime[mode]["metrics"][phase]["n_out"] == (192 if phase=="cold" else 3072)
    prior = [r for r in rows if r["mode"]=="native" and r["phase"]==phase and r["committed_ordinal"]==168]
    assert len(prior)==1 and prior[0]["accepted"]==1 and p["native"]["accepted_proposals_in_round"]==1
    print(phase, "serial margin",diff["serial"],"native margin",diff["native"],"reversal",round(sum(diff.values()),5),"nat")
    assert round(sum(diff.values()),5)==1.19721
print("Replayed 46 scored rows across four diagnostic windows.")
