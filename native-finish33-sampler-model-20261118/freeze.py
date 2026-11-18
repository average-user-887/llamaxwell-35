#!/usr/bin/env python3
"""Freeze rebuilt software and owner-held inputs in the recovered runtime layout."""
from hashlib import file_digest
from pathlib import Path
import os

from model import HERE, OPERATOR, REFERENCE

CORE_INPUTS = (
    "../last-ms-r2-selector-model/FROZEN.sha256",
    "../last-ms-r2-selector-model/INPUTS.sha256",
    "../last-ms-r2-selector-model/model.py",
    "../last-ms-r2-selector-model/release.sh",
    "../last-ms-r2-selector-model/supervise.c",
    "../last-ms-r2-selector-model/check-layout.py",
    "../resume-q2-moe-native-r1/model.py",
    "../night-cache-only/src/tools/server/server-context.cpp",
    "../night-cache-only/src/common/sampling.cpp",
    "../night-cache-only/src/common/common.h",
    "../night-cache-only/libs/libllama-server-impl.so",
    "../night-cache-only/libs/libllama-common.so.0.6.0",
)
RUNTIME = (
    "INPUTS.sha256", "RELEASE.md", "build.sh", "build.json", "build.log", "verify_build.py", "prepare.py",
    "snapshot-policy.inc", "server-context.cpp", "server-context.patch", "libllama-server-impl.so",
    "original-relink.so", "prepare-model.py", "q2_model.py", "model.py", "snapshot-witness.inc", "stride-witness.inc",
    "release.sh", "prepare-layout.py", "supervise.c", "supervise", "check-layout.py", "offline.py",
    "policy-test.py", "clone-screen.cpp", "clone-screen.sh", "clone-screen.log", "sampler-layout.inc", "clone-screen",
)


def freeze(manifest, paths):
    entries = []
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = HERE / path
        with path.open("rb") as source:
            digest = file_digest(source, "sha256").hexdigest()
        entries.append(f"{digest}  {os.path.relpath(path, HERE)}\n")
    (HERE / manifest).write_text("".join(entries))


if __name__ == "__main__":
    inputs = list(CORE_INPUTS)
    inputs.extend(OPERATOR / name for name in ("FROZEN.sha256", "INPUTS.sha256", "libggml-cuda.so"))
    physical = OPERATOR / "receipts/20261118T171658Z-790549"
    inputs.extend(physical / name for name in ("result.json", "status.txt", "observations.sha256"))
    for archive in REFERENCE.values():
        inputs.extend(archive / name for name in ("cold.json", "warm.json", "result.json", "observations.sha256"))
    freeze("INPUTS.sha256", inputs)
    freeze("FROZEN.sha256", (*RUNTIME, "freeze.py"))
