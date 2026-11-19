"""Revalidate sealed owner receipts and export anonymous, per-row numerical evidence."""
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(campaign, serial_path, native_path):
    spec = importlib.util.spec_from_file_location("validator", campaign / "quality80-divergence-r1/model.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    q = v.q
    q.instrument, q.terminal, q.HERE = v.instrument, v.terminal, v.HERE
    pair = q.compare(serial_path, native_path)
    write("divergence.json", pair)
    serial, old = v.revalidate()
    old.pop("source_receipt")
    write("serial-revalidation.json", old)
    native = q.terminal("native", native_path)
    arms = {"native": (native, native_path), "serial": (serial, serial_path)}
    trace, runtime = [], {}
    for mode, ((r, decisions), path) in arms.items():
        full = (path / "server-start.log").read_text()
        need = q.need
        need("offloaded 42/42 layers to GPU" in full and r["accepted_experts"] == 8, "placement")
        shapes = {field: list(map(int, re.findall(rf"llama_context: {field}\s+=\s*(\d+)\b", full)))
                  for field in ("n_seq_max", "n_ctx", "n_ubatch", "n_rs_seq", "n_outputs_max_per_seq")}
        status = dict(line.split("=", 1) for line in (path / "status.txt").read_text().splitlines())
        manifest = dict(line.split("=", 1) for line in (path / "manifest.txt").read_text().splitlines()[:5])
        runtime[mode] = {"mapped_sha256": r["actual_server_software_sha256"], "runtime_image": manifest["image"],
                         "start_epoch": int(manifest["start_epoch"]), "completed_epoch": int(status["completed_epoch"]),
                         "cap_seconds": int(manifest["cap_seconds"]), "runner_exit": int(status["runner_exit"]),
                         "cleanup_exit": int(status["cleanup_exit"]), "layer_offload": [42, 42],
                         "active_experts": r["accepted_experts"], "context_shape": shapes,
                         "archive_text_ids_and_cache_counts_exact_both_phases": True,
                         "metrics": r["metrics"], "sampler_snapshot": r["sampler_snapshot_witness"],
                         "draft_graph": r.get("draft_graph_witness"), "native_routes": r.get("native_routes"),
                         "serial_sparse_widths": r["serial_sparse_widths"], "moe_counters": r["phase_counter_evidence"],
                         "selector_witness": r["selector_execution_witness"], "stride_witness": r["stride_execution_witness"]}
        env = (path / "child-selected-environment.txt").read_text().splitlines()
        numeric = [line.split("=", 1) for line in env if re.fullmatch(r"(?:GGML_CUDA_|LLAMA_)[A-Z0-9_]+=[0-9]+(?:,[0-9]+)*", line)]
        runtime[mode]["observed_numeric_model_flags"] = dict(sorted(numeric))
        runtime[mode]["request_settings"] = {}
        for phase in ("cold", "warm"):
            request = q.read(path / (phase + ".json"))
            runtime[mode]["request_settings"][phase] = {key: request[key] for key in
                ("temperature", "seed", "cache_prompt", "ignore_eos", "n_predict", "prompt_n")}
        buffer = (path / "witness/mv2-private.tsv").read_bytes()
        cut = int((path / "cold-dump-bytes.txt").read_text())
        for ph, block in (("cold", buffer[:cut]), ("warm", buffer[cut:])):
            leading = {info[0][1][ph]["chosen"]: s + "_choice" for s, info in arms.items()}
            for line in block.decode("ascii").splitlines():
                e = q.parse_line(mode, line)
                for i, scores in enumerate(e["scores"]):
                    emitted = i < len(e["ids"])
                    choice = e["ids"][i] if emitted else None
                    trace.append({"phase": ph, "mode": mode, "n_pre": e["n"], "kind": e["kind"],
                                  "accepted": e["accepted"], "row": i, "committed_ordinal": e["n"]+i+1 if emitted else None,
                                  "chosen_rank": list(scores).index(choice)+1 if choice in scores else None,
                                  "chosen_label": leading.get(choice, "other") if emitted else None,
                                  "labels": [leading.get(token, "other") for token in scores],
                                  "logp_descending": list(scores.values())})
    write("runtime.json", runtime)
    (HERE / "rows.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in trace))


if __name__ == "__main__":
    main(*(Path(arg).resolve() for arg in sys.argv[1:]))
