#!/usr/bin/env python3
"""Exercise the binder with synthetic changes to a temporary verified receipt."""
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import model


def rejects(check, reason):
    try:
        check()
    except ValueError as exc:
        assert reason in str(exc)
    else:
        raise AssertionError(f"validator accepted {reason}")


def main():
    model.check()
    with TemporaryDirectory() as scratch:
        out = Path(scratch) / "offline-validator"
        shutil.copytree(model.baseline, out)
        env, _ = model.settings("native")
        (out / "environment.args").write_text("\n".join(env) + "\n")
        (out / "child-selected-environment.txt").write_text("\n".join(env[1::2]) + "\n")

        mapped = out / "runtime-library-hashes.txt"
        records = mapped.read_text()
        for name, digest in model.original.items():
            assert records.count(digest) == 1
            records = records.replace(digest, model.alternate[name], 1)
        mapped.write_text(records)

        start, cold, final = (out / name for name in ("server-start.log", "server-pre-warm.log", "server.log"))
        source_start, source_cold, source_final = (path.read_text() for path in (start, cold, final))
        assert source_cold.startswith(source_start) and source_final.startswith(source_cold)
        startup = source_start + "MTP merge re-eval enabled: synthetic binder test\n"
        first = ("mtp_last_graph pos=80002 rows=3 outputs=1 type=5 kw=3 vw=3 "
                 "qproj=1 q=1 nkv=80128 mask=1 ffn=1 ordered=1\n")
        cold_log = startup + first + source_cold[len(source_start):]
        start.write_text(startup)
        cold.write_text(cold_log)
        final.write_text(cold_log + source_final[len(source_cold):])

        result = model.collect("native", out)
        assert result["accepted_experts"] == 8 and result["draft_graph_witness"]["ordered_cache_writes"]
        assert all(result["actual_server_software_sha256"][key] == digest
                   for key, digest in model.alternate.items())

        name = "libllama.so.0.6.0"
        mapped.write_text(records.replace(model.alternate[name], model.original[name], 1))
        rejects(lambda: model.ready("native", out), "actual executable/implementation/library digest or path differs")
        mapped.write_text(records)
        final.write_text(final.read_text().replace("qproj=1", "qproj=2", 1))
        rejects(lambda: model.collect("native", out), "narrow query witness absent")
    print("CPU validator admitted expected alternate map/graph and refused the old map/wrong query width.")


if __name__ == "__main__":
    main()
