"""Reconstruct the measured sources and owner-local manifests in a private new card."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

HERE = Path(__file__).resolve().parent
PATH_FIELDS = ("N32_FIXTURE", "N72_FIXTURE", "HYBRID_SERIAL")
BASE_FIELDS = ("MODEL_DIRECTORY", "MODEL_BASENAME")


def main(alias_file, sealed_inputs, sealed_frozen, destination):
    values = json.loads(alias_file.read_text())
    if set(values) != {*PATH_FIELDS, *BASE_FIELDS, "MODEL_SHA256"}:
        raise ValueError("incorrect alias fields")
    for field in PATH_FIELDS:
        value = values[field]
        path = PurePosixPath(value)
        if path.is_absolute() or not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or ".." in path.parts:
            raise ValueError(f"{field} must be campaign-relative")
    if any(not re.fullmatch(r"(?!\.{1,2}$)[A-Za-z0-9_.-]+", values[k]) for k in BASE_FIELDS):
        raise ValueError("invalid model directory or basename")
    if not re.fullmatch(r"[a-f0-9]{64}", values["MODEL_SHA256"]):
        raise ValueError("invalid model digest")
    sealed = {name: sha for sha, name in (line.split() for line in sealed_frozen.read_text().splitlines())}
    sources = {}
    for name in ("model.py", "release.sh"):
        content = (HERE / f"{name}.in").read_text()
        content = re.sub(r"@([A-Z0-9_]+)@", lambda match: values[match[1]], content).encode()
        if hashlib.sha256(content).hexdigest() != sealed[name]:
            raise ValueError(f"{name} differs from the owner seal")
        sources[name] = content
    if hashlib.sha256(sealed_inputs.read_bytes()).hexdigest() != sealed["INPUTS.sha256"]:
        raise ValueError("input manifest differs from the owner seal")

    destination.mkdir(mode=0o700, exist_ok=False)
    for name, data in sources.items():
        (destination / name).write_bytes(data)
        (destination / name).chmod(0o700 if name.endswith(".sh") else 0o600)
    shutil.copyfile(sealed_inputs, destination / "INPUTS.sha256")
    with (destination / "FROZEN.sha256").open("w") as output:
        for name in (*sources, "INPUTS.sha256"):
            output.write(f"{sealed[name]}  {name}\n")
    for name in ("INPUTS.sha256", "FROZEN.sha256"):
        (destination / name).chmod(0o600)
    print("Rehydrated two byte-exact measured source files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alias_file", "sealed_inputs", "sealed_frozen", "destination"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
