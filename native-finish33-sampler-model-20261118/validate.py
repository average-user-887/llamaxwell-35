#!/usr/bin/env python3
"""Replay terminal R0/R1 receipts without modifying their recorded outcomes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load_module(Path(__file__).with_name("model.py"), "sampler_r1")


def sha_check(folder, name):
    subprocess.run(["sha256sum", "--check", "--strict", "--quiet", name], cwd=folder, check=True)


def load_original():
    return load_module(model.C / "finish33-sampler/model.py", "sampler_r0")


def verify(mode, receipt):
    receipt = receipt.resolve()
    owner = receipt.parent.parent
    if receipt.parent.name != "receipts" or owner not in (model.HERE, model.C / "finish33-sampler"):
        raise ValueError("expected an owned terminal combined-run receipt")
    sha_check(owner, "FROZEN.sha256")
    sha_check(owner, "INPUTS.sha256")
    sha_check(receipt, "observations.sha256")

    status = dict(line.split("=", 1) for line in (receipt / "status.txt").read_text().splitlines())
    manifest = (receipt / "manifest.txt").read_text().splitlines()
    assert manifest[0] == f"stage={mode}"
    assert "completed_epoch" in status and status["cleanup_exit"] == "0"
    assert any(line.split()[0] == model.n9.hash_file(owner / "FROZEN.sha256") for line in manifest[5:])
    official = model.read(receipt / "sanitized/model-receipt.json")
    rc = int(status["runner_exit"])
    assert (official["mode"], official["runner_exit"], official["cleanup_exit"]) == (mode, rc, 0)

    if rc == 0:
        assert official["refusal"] is None
        result = model.collect(mode, receipt)
        assert result == model.read(receipt / "result.json") == official["validated"]
        disposition = "passed_original_terminal_validator"
    elif rc == 1 and owner == model.C / "finish33-sampler":
        expected = "malformed snapshot witness"
        assert official["refusal"] == expected and official["validated"] is None
        assert (receipt / "run.log").read_text().splitlines()[-1] == f"Q2 model: {expected}"
        original = load_original()
        try:
            original.collect(mode, receipt)
        except ValueError as exc:
            assert str(exc) == expected, exc
        else:
            raise ValueError("R0 refusal did not reproduce")
        result = model.collect(mode, receipt)
        disposition = "passed_r1_timestamp_only_revalidation_of_preserved_r0_refusal"
    else:
        raise ValueError("receipt did not pass and has no authorized timestamp-only recovery")

    sha_check(receipt, "observations.sha256")
    return {
        "mode": mode,
        "raw_runner_exit": rc,
        "raw_cleanup_exit": 0,
        "raw_refusal": official["refusal"],
        "validation_disposition": disposition,
        "raw_observation_manifest_sha256": model.n9.hash_file(receipt / "observations.sha256"),
        "validated": result,
    }


def compare(serial, native):
    observations = {path.resolve(): verify(mode, path) for mode, path in (("serial", serial), ("native", native))}
    old_finished = model.finished
    try:
        model.finished = lambda path: observations[path.resolve()]["validated"]
        result = model.compare(serial, native)
    finally:
        model.finished = old_finished
    result["terminal_validation"] = {
        receipt["mode"]: {key: value for key, value in receipt.items() if key not in ("mode", "validated")}
        for receipt in observations.values()
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--verify", choices=("serial", "native"))
    sub.add_argument("--compare", nargs=2, type=Path, metavar=("SERIAL", "NATIVE"))
    parser.add_argument("receipt", nargs="?", type=Path)
    args = parser.parse_args()
    if args.verify and args.receipt:
        result = verify(args.verify, args.receipt)
    elif args.compare and not args.receipt:
        result = compare(*args.compare)
    else:
        parser.error("supply --verify MODE RECEIPT or --compare SERIAL NATIVE")
    print(json.dumps(result, indent=2, allow_nan=False))
