#!/usr/bin/env python3
"""Adapt the existing typed selector runner to one pinned server implementation."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE.parent
SELECTOR_MODEL = C / "last-ms-r2-selector-model"
Q2_MODEL = C / "resume-q2-moe-native-r1"


def one(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"typed runner preimage mismatch: {before}")
    return source.replace(before, after)


def main():
    q2 = (Q2_MODEL / "model.py").read_text()
    q2 = one(q2, "str(base.SERVER_IMPL):n9.hash_file(base.SERVER_IMPL)",
             "str(base.SERVER_IMPL):SERVER_IMPL_SHA")
    (HERE / "q2_model.py").write_text(q2)

    model = (SELECTOR_MODEL / "model.py").read_text()
    model = one(model, "BASE = C / \"resume-q2-moe-native-r1\"", "BASE = C / \"last-ms-r2-selector-model\"")
    model = one(model, "BASE / \"model.py\"", "HERE / \"q2_model.py\"")
    model = one(model, '    "serial": BASE / "receipts/serial-20261118T142928Z-362150-30056",',
                '    "serial": BASE / "receipts/@OWNER_PRIOR_SELECTOR_SERIAL_RECEIPT@",')
    model = one(model, '    "native": BASE / "receipts/native-20261118T142401Z-347777-15483",',
                '    "native": BASE / "receipts/@OWNER_PRIOR_SELECTOR_NATIVE_RECEIPT@",')
    model = one(model, 'OPERATOR = C / "last-ms-r2-selector"', 'OPERATOR = C / "finish33-r2-stride"')
    model = one(model, 'DSO = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"',
                'DSO = "369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871"')
    model = one(model, 'OLD_DSO = "19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00"',
                'PREVIOUS_DSO = "9e999bcaf04397f267548abdfa3e326b65bee122fb1e24f92be55529f40a335e"')
    model = model.replace("OLD_DSO", "PREVIOUS_DSO")
    model = model.replace("previous_same_mode_19607daf", "previous_same_mode_selector")
    model = model.replace("prior_19607daf", "prior_selector")
    model = model.replace("vs_19607daf", "vs_prior_selector")
    model = model.replace("prior same-mode 19607daf", "prior same-mode selector")
    model = one(model, 'op = read(OPERATOR / "receipts/20261118T151601Z-483850/result.json")',
                'op = read(OPERATOR / "receipts/20261118T171658Z-790549/result.json")')
    model = one(model, """    need(op["physical_gate_pass"] and op["candidate_sha256"] == DSO and op["control_sha256"] == PREVIOUS_DSO,
         "standalone physical gate absent or library differs")""",
                """    need(op["physical_gate_pass"] is False and op["candidate_sha256"] == DSO and
         op["control_sha256"] == PREVIOUS_DSO and op["saved_us_per_mirror"] == {"a": 27.18, "b": 26.72},
         "standalone partial operator receipt or library differs")
    replay = op["independent_candidate_replay"]
    need(set(replay) == {"a-candidate", "b-candidate"} and
         all(row["bitwise_mismatches"] == 0 and row["padded_pitch"] == 2048 and
             row["dense_replay_pitch"] == 1024 for row in replay.values()),
         "standalone parity for both physical pitches absent")""")
    witnesses = (HERE / "snapshot-witness.inc").read_text() + (HERE / "stride-witness.inc").read_text()
    model = one(model, "def settings(stage):\n", witnesses + "\ndef settings(stage):\n")
    model = one(model, 'OPTION + "_CHECK=0"], args',
                'OPTION + "_CHECK=0", "-e", PROBE_OPTION + "=1", "-e", STRIDE_OPTION + "=1",\n'
                '            "-e", SNAPSHOT_OPTION + "=1"], args')
    model = one(model, 'prior.SUP = HERE / "supervise"',
                """prior.SUP = HERE / "supervise"
SERVER_IMPL_SHA = read(HERE / "build.json")["candidate_server_impl_sha256"]
prior.SERVER_IMPL_SHA = SERVER_IMPL_SHA""")
    model = one(model, '    return result\n\ndef finished',
                """    result["sampler_snapshot_witness"] = snapshot_witness(stage, out, result["metrics"])
    result["stride_execution_witness"] = stride_witness(stage, out, result["metrics"], result["selector_execution_witness"])
    return result

def finished""")
    model = one(model, '            "serial_selector_witness": a["selector_execution_witness"]["phases"][phase],',
                """            "serial_selector_witness": a["selector_execution_witness"]["phases"][phase],
            "native_stride_dispatches": b["stride_execution_witness"]["phases"][phase],
            "native_first_kv_pitch_bytes": b["stride_execution_witness"]["first_physical_kv_pitch_bytes"],
            "serial_stride_dispatches": a["stride_execution_witness"]["phases"][phase],
            "native_sampler_snapshots": b["sampler_snapshot_witness"][phase],
            "serial_sampler_snapshots": a["sampler_snapshot_witness"][phase],""")
    model = one(model, '"prior_cuda_sha256": PREVIOUS_DSO,',
                '"prior_cuda_sha256": PREVIOUS_DSO, "candidate_server_impl_sha256": SERVER_IMPL_SHA,')
    model = one(model, 'and value["actual_server_software_sha256"]["libggml-cuda.so"] == PREVIOUS_DSO, "prior model identity")',
                """and value["actual_server_software_sha256"]["libggml-cuda.so"] == PREVIOUS_DSO
             and value["actual_server_software_sha256"]["libllama-server-impl.so"] ==
             read(HERE / "build.json")["original_relink_sha256"], "prior model identity")""")
    model = one(model, "def check():\n", """def check():
    need(n9.hash_file(HERE / "libllama-server-impl.so") == SERVER_IMPL_SHA, "server implementation candidate identity")
""")
    (HERE / "model.py").write_text(model)

    release = (SELECTOR_MODEL / "release.sh").read_text()
    release = one(release, '  "$campaign/resume-q2-moe-native-r1/release.sh" --check',
                  '  "$campaign/last-ms-r2-selector-model/release.sh" --check')
    release = one(release, "overlay=$campaign/last-ms-r2-selector", "overlay=$campaign/finish33-r2-stride")
    release = one(release, '-v "$n1:$n1:ro" -v "$r2:$r2:ro"',
                  '-v "$n1:$n1:ro" -v "$card/libllama-server-impl.so:$n1/libllama-server-impl.so:ro" -v "$r2:$r2:ro"')
    obsolete = """cutoff=$(TZ=Europe/Warsaw date -d '2026-11-18 16:57:00' +%s)
: "${ROOT_GPU_DEADLINE_EPOCH:?supply the benchmark owner cutoff}"
(( deadline <= cutoff )) || { echo 'model arm and cleanup must fit before 16:57 CET' >&2; exit 124; }
"""
    release = one(release, obsolete, ': "${ROOT_GPU_DEADLINE_EPOCH:?supply the benchmark owner cutoff}"\n')
    release = release.replace("nmmoon-r2-top16-model", "nmmoon-finish33-sampler")
    release = release.replace("r2.top16.model.owner", "finish33.sampler.owner")
    release = release.replace("Q2_R2_MODEL_RECEIPT", "FINISH33_SAMPLER_RECEIPT")
    release = release.replace("RESUME_Q2_MODEL_LOCK_FD", "FINISH33_SAMPLER_LOCK_FD")
    release = one(release, '"$card/FROZEN.sha256" "$campaign/model.sha256"',
                  '"$card/FROZEN.sha256" "$card/INPUTS.sha256" "$campaign/model.sha256"')
    (HERE / "release.sh").write_text(release)
    (HERE / "release.sh").chmod(0o700)


if __name__ == "__main__":
    main()
