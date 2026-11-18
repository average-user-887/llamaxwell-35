#!/usr/bin/env python3
"""Rebuild the timestamp-only guard revision with the original binaries."""
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STATIC = ("build.json", "q2_model.py", "libllama-server-impl.so", "supervise",
          "release.sh", "check-layout.py", "INPUTS.sha256")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage-r1.py OWNER_FINISH33_SAMPLER_RUNTIME")
    base = Path(sys.argv[1]).resolve()
    if base.name != "finish33-sampler":
        raise SystemExit("expected the original finish33-sampler runtime")
    subprocess.run(["sha256sum", "--strict", "--check", "--quiet", "FROZEN.sha256"], cwd=base, check=True)
    r1 = base.with_name("finish33-sampler-r1")
    r1.mkdir(mode=0o700)
    for name in STATIC:
        (r1 / name).symlink_to(Path("..") / base.name / name)
    shutil.copyfile(base / "model.py", r1 / "model.py")
    shutil.copyfile(HERE / "parser-r1.patch", r1 / "parser-r1.patch")
    shutil.copyfile(HERE / "validate.py", r1 / "validate.py")
    subprocess.run(["patch", "--silent", str(r1 / "model.py"), str(r1 / "parser-r1.patch")], check=True)
    paths = [*(Path(name) for name in STATIC), Path("model.py"), Path("parser-r1.patch"),
             Path("..") / base.name / "FROZEN.sha256"]
    (r1 / "FROZEN.sha256").write_text("".join(
        f"{sha256((r1 / path).read_bytes()).hexdigest()}  {path}\n" for path in paths))
    (r1 / "VALIDATE.sha256").write_text(f"{sha256((r1 / 'validate.py').read_bytes()).hexdigest()}  validate.py\n")
