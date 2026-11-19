"""Replay published numeric evidence and the immutable compiled-function comparison."""
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def device_functions(name):
    data = gzip.decompress((ROOT / name).read_bytes()).decode()
    pieces = re.split(r"Function : (\S+)\n", data)
    return {name: re.findall(r"/\* (0x[0-9a-f]{16}) \*/", body)
            for name, body in zip(pieces[1::2], pieces[2::2])}

def main():
    result = subprocess.check_output([sys.executable, "-B", str(ROOT / "report.py"), str(ROOT)])
    assert result == (ROOT / "result.json").read_bytes()
    meta = json.loads((ROOT / "build.json").read_text())
    control, treatment = (device_functions(name) for name in ("control.sass.gz", "candidate.sass.gz"))
    assert treatment.keys() - control.keys() == {meta["new_function"]}
    assert all(words == treatment[name] and words for name, words in control.items())
    assert len(control) == meta["original_functions_unchanged"] == 17
    assert sum(map(len, control.values())) == meta["original_machine_words_unchanged"] == 37104
    typed_status = dict(line.split("=", 1) for line in (ROOT / "status.txt").read_text().splitlines())
    assert typed_status["runner_exit"] == typed_status["cleanup_exit"] == "0"
    print("PASS: replayed all materiality intervals, numeric words, physical status and prior compiled functions")

if __name__ == "__main__":
    main()
