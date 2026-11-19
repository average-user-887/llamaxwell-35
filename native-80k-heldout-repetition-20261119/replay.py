"""Reconcile the published score, route and recovery evidence without an owner fixture."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTERS = ("row0", "row1", "combined", "skip", "q2_sparse", "q2_stride")


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    pair = read("paired-scores.json")
    setup = read("runtime.json")
    failure = read("original-serial-failure.json")
    assert pair["technical_pass"] and setup["same_preflight_model_binding"]
    arms = setup["arms"]
    for key in ("image_sha256", "actual_server_software_sha256", "effective_numeric_flags",
                "server_context_tokens", "gpu_layers_offloaded", "accepted_experts", "one_slot"):
        assert arms["serial"][key] == arms["native"][key]
    for arm in arms.values():
        assert arm["technical_pass"] and arm["request_order"] == ["n32", "n72"]
        assert arm["runner_exit"] == arm["cleanup_exit"] == 0
        assert 0 < arm["completed_epoch"] - arm["start_epoch"] == arm["bounded_wall_s"] <= arm["cap_seconds"]
        assert arm["deadline_epoch"] - arm["start_epoch"] == arm["cap_seconds"] == 600
        assert arm["accepted_experts"] == 8 and arm["gpu_layers_offloaded"] == 42 and arm["one_slot"]
        assert arm["server_context_tokens"] == 84224
        assert arm["start_admission"]["util_pct"] == 0
        witness = arm["counter_witnesses"]
        for case, first, second in (("n32", "start", "cold"), ("n72", "cold", "warm")):
            delta = {key: witness[second][key] - witness[first][key] for key in COUNTERS}
            assert delta == arm["route"]["physical_deltas"][case]

    for case, prompt_n, count, offered, late, late_total, repeats, lines in (
        ("n32", 80000, 1722, 861, 5, 38, 20, 149),
        ("n72", 32768, 1087, 543, 0, 21, 8, 81),
    ):
        row = pair["cases"][case]
        assert row["paired_parity"] == {
            "all_ids_equal": True, "text_equal": True, "first_differing_id_1_based": None,
        }
        for mode in arms:
            completion = row[mode]
            scope = arms[mode]
            assert completion["prompt_tokens"] == prompt_n and completion["cached_prompt_tokens"] == 0
            assert completion["completion_tokens_including_eos"] == count and completion["stop_type"] == "eos"
            assert completion["request_params"] == row["serial"]["request_params"]
            assert completion["actual_sampling"] == row["serial"]["actual_sampling"]
            assert completion["request_params"]["cache_prompt"] is False
            assert all(completion["quality"][key] for key in
                       ("eos", "semantic_pass", "exact_expected_ast", "exact_expected_text"))
            assert completion["archive_parity"] == row["paired_parity"]
            assert scope["start_epoch"] <= completion["start_epoch_ns"] / 1e9
            assert completion["start_epoch_ns"] < completion["end_epoch_ns"] <= (1 + scope["completed_epoch"]) * 1e9
            assert 0 < completion["server_prompt_ms"] + completion["server_decode_ms"] <= completion["wall_s"] * 1000
            rpt = completion["repetition"]
            assert rpt == completion["expected_source_repetition"]
            assert (rpt["final_quarter_normalized_duplicates"], rpt["final_quarter_lines"],
                    rpt["duplicate_normalized_lines"], rpt["nonblank_lines"]) == (late, late_total, repeats, lines)
            assert rpt["repeated_24_word_windows"] == rpt["trailing_comment_only_lines"] == rpt["malformed_characters"] == 0
            if mode == "serial":
                assert completion["draft_offered"] == completion["draft_accepted"] == 0
                assert not any(scope["route"]["physical_deltas"][case].values())
            else:
                assert completion["draft_offered"] == completion["draft_accepted"] == offered
                physical = scope["route"]["physical_deltas"][case]
                assert physical["combined"] == physical["skip"] == 40 * offered
                assert physical["q2_sparse"] == 10 * offered
                assert physical["q2_stride"] == (10 * offered if case == "n32" else 0)

    assert arms["serial"]["route"]["merged_80k_graph"] is None
    assert arms["native"]["route"]["merged_80k_graph"] == {
        "position": 80002, "kv_write_rows": 3, "key_width": 80128, "query_rows": 1,
    }
    assert failure["runner_exit"] == 2 and failure["cleanup_exit"] == 0
    assert failure["original_observation_manifest_verified"] and failure["second_request_sent"] is False
    assert failure["awaited_witness"] == "n32.json" and failure["observed_witness"] == "cold.json"
    assert failure["correction"] == {"startup": "start.json", "n32": "cold.json", "n72": "warm.json"}
    assert set(failure["counter_witnesses"]) == {"start", "cold"}
    covered = failure["completed_request"]
    assert covered["quality"]["semantic_pass"] and covered["quality"]["eos"]
    assert covered["prompt_tokens"] == 80000 and covered["cached_prompt_tokens"] == 0
    assert covered["completion_tokens_including_eos"] == 1722
    assert covered["repetition"] == pair["cases"]["n32"]["serial"]["repetition"]
    print("PASS: two exact scored edits, paired timing and repetition, route counters, and the original witness refusal.")


if __name__ == "__main__":
    main()
