"""Recreate the original byte strings using the owner's build mount and sealed fixture."""
import argparse
import hashlib
import json
import re
from pathlib import Path

base = Path(__file__).resolve().parent
arg = argparse.ArgumentParser()
arg.add_argument("build_source_mount", type=Path)
arg.add_argument("earlier_q8_rows", type=Path)
arg.add_argument("output", type=Path)
args = arg.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
shape, = re.findall(r"^N34_CALL stage=badpair at=2004 graphs=1 in=2 out=2 input0=(\d+) input_last=(\d+) ", args.earlier_q8_rows.read_text(), re.M)
pins = json.loads((base / "actual-source-hashes.json").read_text())
for artifact in sorted(base.glob("*.in")):
    name = artifact.name.removesuffix(".in")
    raw = artifact.read_bytes().replace(b"@CUDA_BUILD_SOURCE_MOUNT@", str(args.build_source_mount).encode())
    raw = raw.replace(b"@PENDING_TARGET_ID@", shape[0].encode())
    raw = raw.replace(b"@WRONG_TARGET_ID@", shape[1].encode())
    if hashlib.sha256(raw).hexdigest() != pins[name]:
        raise ValueError(f"{name} differs from source used in the physical measurement")
    (args.output / name).write_bytes(raw)
print("All five hydrated files match their measured hashes.")
