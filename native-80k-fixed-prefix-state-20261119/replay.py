"""Recreate the portable result from the anonymous measured transcript."""
import json
from pathlib import Path
from report import summarize

HERE = Path(__file__).resolve().parent
status = json.loads((HERE / "runtime.json").read_text())
actual = summarize((HERE / "witness.txt").read_text(), status["runner_exit"], status["cleanup_exit"])
expected = json.loads((HERE / "result.json").read_text())
if actual != expected:
    raise SystemExit("Anonymous witness and result differ")
if not actual["executed_required_paths"]:
    raise SystemExit("The anonymous witness lacks physical coverage")
print("Verified the observed graph, state and score result from the anonymous witness.")
