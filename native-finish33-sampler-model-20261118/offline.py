"""Exercise the frozen validator on copies of exact historical receipts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import model


def stage(root, mode):
    ref = model.REFERENCE[mode]
    out = root / mode
    out.mkdir()
    for source in ref.iterdir():
        if source.name not in ("result.json", "observations.sha256", "witness"):
            (out / source.name).symlink_to(source)
    env, _ = model.settings(mode)
    for name, content in (("environment.args", env), ("child-selected-environment.txt", env[1::2])):
        path = out / name
        path.unlink()
        path.write_text("\n".join(content) + "\n")

    mappings = model.prior.base.maps(ref / "runtime-library-hashes.txt")
    mappings[str(model.prior.base.SERVER_IMPL)] = model.SERVER_IMPL_SHA
    mappings["/candidate/libggml-cuda.so"] = model.DSO
    path = out / "runtime-library-hashes.txt"
    path.unlink()
    path.write_text("".join(f"{sha}  {name}\n" for name, sha in mappings.items()))
    supervisor = out / "supervisor.sha256"
    supervisor.unlink()
    supervisor.write_text(model.n9.hash_file(model.prior.SUP) + "  /proc/1/exe\n")
    (out / "witness").mkdir()
    for name in ("start", "cold", "warm"):
        old = model.read(ref / "witness" / f"{name}.json")
        (out / "witness" / f"{name}.json").write_text(json.dumps(dict(old, q2_stride=old["q2_sparse"])) + "\n")
    (out / "witness/child.pid").symlink_to(ref / "witness/child.pid")
    samples = []
    prefix = (ref / "server-pre-warm.log").read_text()
    markers = ""
    if mode == "native":
        markers = ("q2_r2_probe4_dispatch width=80128 plans=4 blocks=313 tiles=79\n"
                   "q2_r2_stride_dispatch width=80128 k_token_bytes=1024 v_token_bytes=1024 splits=12 nu=4 threads=512 grid=24\n")
    offered = rejected = 0
    for phase, name in zip(("cold", "warm"), ("server-pre-warm.log", "server.log")):
        timing = model.read(ref / f"{phase}.json")["timings"]
        proposals = timing.get("draft_n") or 0
        accepted = timing.get("draft_n_accepted") or 0
        offered += proposals
        rejected += proposals - accepted
        samples.append(f"sampler_snapshot: elided={offered} elided_rejected={rejected} kept=0 restored=0\n")
        path = out / name
        path.unlink()
        raw = (ref / name).read_text()
        assert raw.startswith(prefix)
        previous = samples[0] if phase == "warm" else ""
        path.write_text(prefix + markers + previous + raw[len(prefix):] + samples[-1])

    (out / "result.json").write_text(json.dumps(model.collect(mode, out)) + "\n")
    return out


def refuses(action, expected):
    try:
        action()
    except ValueError as exc:
        assert expected in str(exc), exc
    else:
        raise AssertionError(f"accepted bad input: {expected}")


with TemporaryDirectory(dir=model.HERE) as tmp:
    root = Path(tmp)
    serial, native = (stage(root, mode) for mode in ("serial", "native"))
    pair = model.compare(serial, native)
    assert pair["technical_pass"] and not pair["warm_same_build_25_percent_gate"]
    assert pair["candidate_server_impl_sha256"] == model.SERVER_IMPL_SHA
    assert pair["measurements"]["warm"]["native_saved_ms_per_round_vs_prior_selector"] == 0
    final = native / "server.log"
    original = final.read_text()
    final.write_text(original.replace("elided_rejected=225 kept=0 restored=0",
                                      "elided_rejected=225 kept=0 restored=1"))
    refuses(lambda: model.collect("native", native), "sampler snapshots")
    final.write_text(original.replace("elided=1743", "elided=1742"))
    refuses(lambda: model.collect("native", native), "sampler snapshots")
    final.write_text(original)
    final.write_text(original.replace("k_token_bytes=1024", "k_token_bytes=4096"))
    cold = native / "server-pre-warm.log"
    original_cold = cold.read_text()
    cold.write_text(original_cold.replace("k_token_bytes=1024", "k_token_bytes=4096"))
    refuses(lambda: model.collect("native", native), "stride pitch outside physical")
    final.write_text(original)
    cold.write_text(original_cold)
    witness = native / "witness/warm.json"
    counts = model.read(witness)
    witness.write_text(json.dumps(dict(counts, q2_stride=counts["q2_stride"] - 1)) + "\n")
    refuses(lambda: model.collect("native", native), "fixed-stride engagement")
    witness.write_text(json.dumps(counts) + "\n")
    maps = native / "runtime-library-hashes.txt"
    maps.write_text(maps.read_text().replace(model.SERVER_IMPL_SHA, "0" * 64))
    refuses(lambda: model.collect("native", native), "actual executable")
print("Archived exact-output comparison replay passed both arms and refused corrupted restore, bypass, stride and mapping evidence.")
