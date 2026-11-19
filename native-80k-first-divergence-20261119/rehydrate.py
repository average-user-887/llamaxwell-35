"""Rehydrate published source templates in an empty private directory for owner replay."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ALIASES = {"HYBRID_NATIVE", "HYBRID_SERIAL", "SERIAL_SCORE", "MODEL_SHA256", "MODEL_DIRECTORY", "MODEL_BASENAME"}


def main(aliases, destination):
    values = json.loads(aliases.read_text())
    assert set(values) == ALIASES
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", val) for val in values.values())
    digests = json.loads((HERE / "source-identities.json").read_text())
    outputs = {}
    for name, metadata in digests.items():
        text = (HERE / (name + ".in")).read_text()
        for k, val in values.items():
            text = text.replace("@" + k + "@", val)
        assert hashlib.sha256(text.encode()).hexdigest() == metadata["original_source_sha256"]
        outputs[metadata["output"]] = text
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, contents in outputs.items():
        path = destination / name
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.write_text(contents)
        path.chmod(0o700 if path.name.endswith(".sh") else 0o600)
    print("Rehydrated four byte-exact sources.")


if __name__ == "__main__":
    main(*(Path(x) for x in sys.argv[1:]))
