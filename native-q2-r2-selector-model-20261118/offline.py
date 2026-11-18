"""Replay accepted archived results with synthetic new runtime identity and counts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import model as model

def write(path, settings):
    path.write_text(json.dumps(settings) + "\n")

def staged(root, mode):
    ref = model.REFERENCE[mode]
    out = root / mode
    out.mkdir()
    witness = out / "witness"
    witness.mkdir()
    for name in ("admission.txt", "cold.json", "warm.json", "server-start.log"):
        (out / name).symlink_to(ref / name)
    marker = "q2_r2_top16_dispatch n_q=2 n_kv=80128 nkvh=2 plans=4\n"
    start_len = len((ref / "server-start.log").read_text())
    for name in ("server-pre-warm.log", "server.log"):
        text = (ref / name).read_text()
        (out / name).write_text(text[:start_len] + (marker if mode == "native" else "") + text[start_len:])
    env, args = model.settings(mode)
    for name, data in (("environment.args", env), ("server.args", args),
                       ("child-selected-environment.txt", env[1::2]),
                       ("child-server.args", ["/candidate/llama-server", *args])):
        (out / name).write_text("\n".join(data) + "\n")
    (witness / "child.pid").write_text("99\n")
    aliases = model.prior.base.maps(ref / "runtime-library-hashes.txt")
    server_exe = next(key for key in aliases if key.startswith("/proc/"))
    aliases["/proc/99/exe"] = aliases.pop(server_exe)
    aliases["/candidate/libggml-cuda.so"] = model.DSO
    (out / "runtime-library-hashes.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in aliases.items()))
    (out / "supervisor.sha256").write_text(model.n9.hash_file(model.prior.SUP) + "  /proc/1/exe\n")
    count = dict(child=99, row0=0, row1=0, combined=0, skip=0, q2_sparse=0)
    write(witness / "start.json", count)
    for phase in ("cold", "warm"):
        drafts = model.read(out / f"{phase}.json")["timings"].get("draft_n") or 0
        count["combined"] += 40 * drafts
        count["skip"] += 40 * drafts
        count["q2_sparse"] += 10 * drafts
        write(witness / f"{phase}.json", count)
    write(out / "result.json", model.collect(mode, out))
    (out / "status.txt").write_text("runner_exit=0\ncleanup_exit=0\n")
    return out

def refuses(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"accepted damaged input: {message}")

with TemporaryDirectory(dir=model.HERE) as tmp:
    root = Path(tmp)
    serial, native = (staged(root, stage) for stage in ("serial", "native"))
    comparison = model.compare(serial, native)
    assert comparison["technical_pass"] and not comparison["warm_same_build_25_percent_gate"]
    assert comparison["measurements"]["warm"]["native_saved_ms_per_round_vs_19607daf"] == 0.0
    metrics = model.read(native / "result.json")["metrics"]
    log = native / "server.log"
    good = log.read_text()
    log.write_text(good.replace("q2_r2_top16_dispatch", "removed"))
    refuses(lambda: model.physical_witness("native", native, metrics), "fast-selector launch missing")
    log.write_text(good.replace("n_kv=83200 nkvh=2 nsplit=12", "n_kv=83456 nkvh=2 nsplit=12"))
    refuses(lambda: model.physical_witness("native", native, metrics), "outside the approved selector")
    log.write_text(good)
    warm = native / "witness/warm.json"
    data = model.read(warm)
    write(warm, dict(data, q2_sparse=data["q2_sparse"] - 1))
    refuses(lambda: model.physical_witness("native", native, metrics), "ten per offered verifier")
    write(warm, data)
    env = native / "child-selected-environment.txt"
    env.write_text(env.read_text().replace(model.OPTION + "=1", model.OPTION + "=0"))
    refuses(lambda: model.prior.ready("native", native), "child flags differ")
print("Offline historical replay passed for both modes; refused missing branch, out-of-range width, wrong count and disabled flag.")
