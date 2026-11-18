#!/usr/bin/env python3
"""Verify the candidate implementation's exact preimage and dynamic ABI."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "night-cache-only/libs/libllama-server-impl.so"
CANDIDATE = HERE / "libllama-server-impl.so"
ORIGINAL_SHA = "1138515898887c6ec6be01a210cfb59bb8e1338ccf6eeef1d032ffd4fdb22b65"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exports(path):
    output = subprocess.check_output(["nm", "-D", "--defined-only", path], text=True)
    return sorted(" ".join(line.split()[1:]) for line in output.splitlines())


def dependencies(path):
    output = subprocess.check_output(["readelf", "-d", path], text=True)
    return re.findall(r"\((NEEDED|SONAME|RUNPATH)\).*?\[([^\]]+)\]", output)


def main():
    if digest(BASE) != ORIGINAL_SHA or digest(HERE / "original-relink.so") != ORIGINAL_SHA:
        raise ValueError("original server implementation or its isolated relink differs")
    original_exports = exports(BASE)
    original_deps = dependencies(BASE)
    if exports(CANDIDATE) != original_exports or dependencies(CANDIDATE) != original_deps:
        raise ValueError("server implementation dynamic ABI or loader identity changed")
    candidates = {
        "scope": "exact original relink and equal dynamic symbol type/name list, NEEDED, SONAME and RUNPATH",
        "original_relink_sha256": ORIGINAL_SHA, "candidate_server_impl_sha256": digest(CANDIDATE),
        "source_patch_sha256": digest(HERE / "server-context.patch"),
        "dynamic_exports": len(original_exports),
        "dynamic_export_list_sha256": hashlib.sha256("\n".join(original_exports).encode()).hexdigest(),
        "dynamic_dependencies": original_deps,
    }
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
