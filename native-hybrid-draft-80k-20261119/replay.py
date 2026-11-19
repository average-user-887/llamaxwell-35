"""Independently reconcile safe typed receipt boundaries and published metrics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def load_trace(index, mode):
    artifact = ROOT / index[mode]["file"]
    contents = artifact.read_bytes()
    rows = [json.loads(line) for line in contents.splitlines()]
    assert hashlib.sha256(contents).hexdigest() == index[mode]["trace_sha256"]
    assert all(row["arm"] == mode for row in rows)
    assert Counter(row["event"] for row in rows) == index[mode]["events"]
    assert rows[-1]["event"] == "owned_run_outcome"
    assert rows[-1]["status"]["completed_epoch"] == index[mode]["completed_epoch"]
    return rows


def main():
    index = read("trace-index.json")
    native, serial = (load_trace(index, mode) for mode in ("native", "serial"))
    assert index["owned_arms_did_not_overlap"] and index["arm_order"] == ["native", "serial"]
    assert index["native"]["completed_epoch"] <= index["serial"]["start_epoch"]
    assert len([row for row in native if row["event"] == "draft_final_query_first_dispatch"]) == 1
    assert not any(row["event"] == "draft_final_query_first_dispatch" for row in serial)
    rep = read("repetition.json")
    if not (ROOT / "pair-audit.json").exists():
        assert not (ROOT / "same-build-pair.json").exists()
        assert serial[-1]["runner_exit"] or serial[-1]["cleanup_exit"]
        print("PASS: replayed terminal arm traces and available repetition metrics")
        return
    audit, comparison = (read(name) for name in ("pair-audit.json", "same-build-pair.json"))
    assert native[-1]["runner_exit"] == serial[-1]["runner_exit"] == 0
    for phase, count in (("cold", 192), ("warm", 3072)):
        observations = [next(row for row in trace
                             if row["event"] == "observed_response_metrics" and row["phase"] == phase)
                        for trace in (serial, native)]
        assert len([row for row in native if row.get("phase") == phase and
                    row["event"] == "observed_stream_output_ordinal"]) == count
        serial_t, native_t = (row["server_timings"] for row in observations)
        assert serial_t["predicted_n"] == native_t["predicted_n"] == count
        assert serial_t["cache_n"] == native_t["cache_n"] == (0 if phase == "cold" else 79996)
        assert serial_t["prompt_n"] == native_t["prompt_n"] == (80000 if phase == "cold" else 4)
        gain = 100 * (serial_t["predicted_ms"] / native_t["predicted_ms"] - 1)
        assert abs(gain - audit["metrics"][phase]["decode_gain_pct"]) < 1e-10
        assert abs(gain - comparison["measurements"][phase]["same_build_native_over_serial_pct"]) < 1e-10
        remaining = max(0, (native_t["predicted_ms"] - count * 10) / native_t["draft_n"])
        assert abs(remaining - audit["metrics"][phase]["additional_ms_per_offer_to_100"]) < 1e-12
        parity = rep["phases"][phase]["parity"]
        assert parity["hybrid_native_vs_clean_split_reference"]["full_text_equal"]
        assert parity["hybrid_serial_vs_clean_serial_reference"]["full_text_equal"]
        assert parity["hybrid_native_vs_hybrid_serial"]["first_id_difference_1_based"] == 170
    print("PASS: replayed full typed traces and checked paired performance and repetition summaries")


if __name__ == "__main__":
    main()
