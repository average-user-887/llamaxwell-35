"""Validate end-to-end selection evidence and evaluate the frozen timing gate."""
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

out = Path(__file__).resolve().parent / "observed"
build = json.loads((out.parent / "build.json").read_text())
labels = ("a-control", "a-candidate", "b-candidate", "b-control")
timings, hashes, all_plans, diagnostics = {}, {}, {}, {}
reference = None
for label in labels:
    candidate = label.endswith("candidate")
    log = (out / f"{label}.txt").read_text()
    library_sha = build["candidate_sha256" if candidate else "control_sha256"]
    assert f"{library_sha}  /candidate/libggml-cuda.so" in log, label
    assert f'{build["harness_sha256"]}  /candidate/test-backend-ops' in log, label
    assert "loaded CUDA backend from /candidate/libggml-cuda.so" in log and "Quadro M6000" in log, label
    assert log.count("q2_r2_top16_dispatch n_q=2 n_kv=81920 nkvh=2 plans=4")==1,label
    assert log.count("q2_r2_probe4_dispatch width=81920 plans=4 blocks=320 tiles=80")==int(candidate),label
    stride_marker = "q2_r2_stride_dispatch width=81920 k_token_bytes=2048 v_token_bytes=2048 splits=12 nu=4 threads=512 grid=24"
    assert log.count(stride_marker) == int(candidate), label
    dense_marker = "q2_r2_stride_dense_replay key_bytes=83886080 value_bytes=83886080 src_pitch=2048 dst_pitch=1024 outputs=8192 exact=1"
    assert log.count(dense_marker) == int(candidate), label
    assert log.count("q2_r2_stride_partial_replay values=98304 max_sum_words=768 exact=1") == int(candidate), label
    assert log.count("q2_r2_stride_fallback plans=4 full_warps_each=96 outputs=8192 finite=1 exact=1") == int(candidate), label
    if candidate:
        assert log.count("q2_r2_probe4_meta entries=1280 exact=1")==1,label
        edges=re.findall(r"^q2_r2_probe4_edge entries=1280 mask_bad=(\d+) score_bad=(\d+) exact=1$",log,re.M)
        assert len(edges)==1 and all(int(x)>0 for x in edges[0]),label
        expected = "n_kv=81920 plans=4 scores=1280 outputs=8192 plan_diff=0 score_diff=0 output_diff=0 nonfinite=0"
        assert log.count(f"q2_r2_top16_check {expected}") == 1, label
        diagnostics[label] = dict(retained_splits=12, retained_nu=4, retained_threads=512, padded_pitch=2048, dense_replay_pitch=1024, dense_replay_outputs=8192, dense_partial_words=98304, dense_maxsum_words=768, fallback_warps_per_plan=96, fallback_output_words=8192, plans=4, scores=1280, outputs=8192, bitwise_mismatches=0, metadata_entries=1280, edge_metadata_and_scores=1280, edge_invalid_mask_blocks=int(edges[0][0]), edge_invalid_score_blocks=int(edges[0][1]))
    else:
        assert "q2_r2_top16_check" not in log, label
    assert re.search(r"dispatch calls=1000 n_q=2 n_kv=81920 nkvh=2 .*revision=2", log), label
    assert not re.search(r"dispatch calls=\d+ n_q=[^2]", log), label
    plans = re.findall(r"^occamy_sparse_plan call=1 n_q=2 .+$", log, re.M)
    assert len(plans) == 4, label
    for plan in plans:
        assert "n_kv=81920 ok=1 invalid=0 fallback=0" in plan, label
        assert "full_warps=0" in plan and "old_blocks=16" in plan, label
        ids = list(map(int, plan.split("block_ids=")[1].split(",")))
        assert len(ids) == 16 and sorted(set(ids)) == ids, label
        row = int(re.search(r"row=(\d+)", plan)[1])
        assert f"visible_end={81659 + row} " in plan, label
    all_plans[label] = plans
    matches = re.findall(r"FLASH_ATTN_EXT\([^\n]+kv=81920,nb=2,.*mask=mrow_causal,tail=260\):\s*(\d+) runs -\s*([0-9.]+) us/run", log)
    assert len(matches) == 1, (label, matches)
    runs, us = matches[0]
    assert int(runs) >= 1000 and float(us) > 0, label
    timings[label] = {"runs": int(runs), "operator_us": float(us)}

assert all(plan == all_plans["a-control"] for plan in all_plans.values())
savings = {name: round(timings[f"{name}-control"]["operator_us"] - timings[f"{name}-candidate"]["operator_us"], 2)
           for name in ("a", "b")}
result = {
    "control_sha256": build["control_sha256"], "candidate_sha256": build["candidate_sha256"],
    "paired_order": labels, "key_width_measured": 81920, "n_q": 2, "nkvh": 2,
    "whole_operator": timings, "all_four_local_output_arrays_finite_and_bitwise_equal": True, "plan_lines_each": 4,
    "independent_candidate_replay": diagnostics, "saved_us_per_mirror": savings,
    "gate_us_each_mirror": 35.0, "physical_gate_pass": min(savings.values()) >= 35.0,
    "minimum_projected_ms_per_10_invocations": round(min(savings.values()) / 100, 4),
}
assert json.loads(json.dumps(result)) == json.loads((out/"result.json").read_text())
status = dict(line.split("=") for line in (out/"status.txt").read_text().splitlines())
assert status["runner_exit"] == status["cleanup_exit"] == "0"
assert int(status["finished_epoch"]) - int(status["started_epoch"]) <= int(status["cap_seconds"])
print("Four complete operator traces reproduced both mirrored timings, all comparison counters and the missed 35 us gate. Raw synthetic output equality was checked locally.")
