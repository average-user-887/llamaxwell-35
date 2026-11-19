"""Pair frozen IFEval prompts and score them with the installed IFEval evaluator."""
import argparse
from hashlib import file_digest, sha256
from http.client import HTTPConnection, HTTPException
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import time

ROOT = Path(__file__).resolve().parent
LOCK = ROOT.parents[1] / "gpu-bench.lock"
HOME = Path.home()
SITE = HOME / ".local/share/lab-evals/venv/lib/python3.12/site-packages"
REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
SCORER_REVISION = "0c495b2f95155e8b10acb919ae283bfb4d5be6e2"
INSPECT_EVALS_REVISION = "32b79a244f17ecf54fa1186d30f33dc10143c10f"
DATA = HOME / ".cache/huggingface/hub/datasets--google--IFEval/snapshots" / REVISION / "ifeval_input_data.jsonl"
NLTK = HOME / ".cache/nltk_data"
DATA_SHA256 = "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
SALT = "llamaxwell-35-ifeval-r2-v1"
OPTIONS = {
    "seed": 7,
    "temperature": 0.0,
    "max_tokens": 1024,
    "stream": False,
    "cache_prompt": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
REQUEST_SECONDS = 90
WALL_LIMITS = {"smoke": 560, "full": 50_400}
SHARED = ("model", "freeze_sha256", "program_sha256", "binary_sha256", "weights_sha256",
          "template_sha256", "profile_pair_sha256")
SHA = re.compile(r"[0-9a-f]{64}")
MODEL = re.compile(r"(?:qwen|llamaxwell)[a-zA-Z0-9._-]{0,92}", re.IGNORECASE)
OUTCOMES = ("both_pass", "both_fail", "on_only", "off_only", "missing")


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def package_pin(name):
    dist = metadata.distribution(name)
    return json.loads(dist.read_text("direct_url.json"))["vcs_info"]["commit_id"]


def load_examples():
    from inspect_evals.ifeval.ifeval import IFEVAL_DATASET_REVISION, record_to_sample
    from instruction_following_eval.evaluation import InputExample

    if IFEVAL_DATASET_REVISION != REVISION or digest(DATA) != DATA_SHA256:
        raise ValueError("The cached Google IFEval snapshot differs from the pinned input")

    examples = {}
    for line in DATA.read_text().splitlines():
        record = json.loads(line)
        sample = record_to_sample(record)
        meta = sample.metadata
        if sample.id in examples or type(sample.id) is not int or sample.input != meta["prompt"]:
            raise ValueError("Corrupt or unsupported IFEval input")
        examples[sample.id] = InputExample(sample.id, meta["instruction_id_list"],
                                          meta["prompt"], meta["kwargs"])
    if len(examples) != 541 or sum(len(x.instruction_id_list) for x in examples.values()) != 834:
        raise ValueError("The IFEval snapshot must contain 541 prompts and 834 instructions")
    return examples


def build_manifest(examples):
    if metadata.version("inspect_ai") != "0.3.260":
        raise ValueError("The installed Inspect AI version changed")
    if package_pin("inspect_evals") != INSPECT_EVALS_REVISION:
        raise ValueError("The installed Inspect Evals source changed")
    if package_pin("instruction_following_eval") != SCORER_REVISION:
        raise ValueError("The installed IFEval scorer source changed")

    paths = list((SITE / "instruction_following_eval").glob("*.py"))
    paths += [SITE / "inspect_evals/ifeval/ifeval.py", SITE / "inspect_evals/ifeval/eval.yaml"]
    paths += list((NLTK / "tokenizers/punkt_tab/english").glob("*"))
    paths += [NLTK / "tokenizers/punkt/english.pickle"]
    paths += list((SITE / "langdetect/profiles").glob("*"))
    full = sorted(examples)
    smoke = sorted(full, key=lambda key: (sha256(f"{SALT}\0smoke\0{key}".encode()).digest(), key))[:6]
    return {
        "dataset": "google/IFEval", "split": "train", "revision": REVISION,
        "dataset_sha256": DATA_SHA256,
        "inspect_evals_revision": INSPECT_EVALS_REVISION, "scorer_revision": SCORER_REVISION,
        "versions": {name: metadata.version(name) for name in (
            "inspect_ai", "inspect_evals", "instruction_following_eval", "nltk", "langdetect", "regex"
        )},
        "files": {str(path.relative_to(HOME)): digest(path) for path in sorted(paths) if path.is_file()},
        "salt": SALT, "stages": {"full": full, "smoke": smoke},
        "instruction_counts": {stage: sum(len(examples[key].instruction_id_list) for key in ids)
                               for stage, ids in (("full", full), ("smoke", smoke))},
        "options": OPTIONS, "request_seconds": REQUEST_SECONDS, "wall_seconds_max": WALL_LIMITS,
        "max_concurrency": 1, "max_requests_per_case": 1,
    }


def verify_manifest(examples):
    frozen = read_json(ROOT / "freeze.json")
    if frozen != build_manifest(examples):
        raise ValueError("Pinned data, package, case order or run settings differ from the freeze")
    return frozen


def setup_scorer():
    import langdetect
    import nltk

    # The fork's resource helper can download. Resolve English data only from the verified cache.
    nltk.data.path[:] = [str(NLTK)]
    nltk.data.find("tokenizers/punkt/english.pickle")
    nltk.data.find("tokenizers/punkt_tab/english/")
    langdetect.DetectorFactory.seed = 7


def score(example, answer):
    from instruction_following_eval.evaluation import test_instruction_following

    scores = {}
    for mode in ("strict", "loose"):
        random.seed(7)
        result = test_instruction_following(example, answer, strict=mode == "strict")
        scores[mode] = list(result.follow_instruction_list)
    return scores


def local_port(endpoint):
    match = re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1", endpoint)
    if not match or int(match[1]) > 65535:
        raise ValueError("Expected explicit numeric loopback http://127.0.0.1:PORT/v1")
    return int(match[1])


def validate_admission(fd, lock_fd, stage, arm):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Expected a private same-UID supervisor admission descriptor")
    raw = os.pread(fd, 16_385, 0)
    if not 0 < len(raw) <= 16_384:
        raise ValueError("Admission record is empty or too large")
    admission = json.loads(raw)
    if admission.get("arm") != arm or admission.get("stage") != stage or admission.get("r2_same_build") is not True:
        raise ValueError("Admission arm, stage or same-build attestation mismatches")
    if admission.get("freeze_sha256") != digest(ROOT / "freeze.json") or admission.get("program_sha256") != digest(__file__):
        raise ValueError("Admission does not bind this program and freeze")
    if any(not isinstance(admission.get(field), str) or not SHA.fullmatch(admission[field]) for field in SHARED[1:]):
        raise ValueError("Admission must provide all identity hashes")
    if not isinstance(admission.get("model"), str) or not MODEL.fullmatch(admission["model"]):
        raise ValueError("Admission must provide a local Qwen/LlamaXwell alias")
    wall = admission.get("wall_seconds")
    if type(wall) is not int or not 1 <= wall <= WALL_LIMITS[stage]:
        raise ValueError("Admission wall budget exceeds the frozen stage cap")
    local_port(admission["endpoint"])

    opened = os.fstat(lock_fd)
    expected = LOCK.stat()
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Expected the existing campaign GPU lock descriptor")
    locks = Path(f"/proc/self/fdinfo/{lock_fd}").read_text().splitlines()
    if not any("FLOCK" in line and "WRITE" in line for line in locks if line.startswith("lock:")):
        raise ValueError("Campaign GPU lock descriptor is not exclusively flocked")
    return admission


def request_body(example, alias):
    return {"model": alias, "messages": [{"role": "user", "content": example.prompt}], **OPTIONS}


def parse_completion(raw, alias):
    reply = json.loads(raw)
    if not isinstance(reply, dict) or reply.get("model") != alias:
        raise ValueError("Response model alias mismatch")
    choice, = reply["choices"]
    if choice["finish_reason"] not in {"stop", "length"}:
        raise ValueError("Unexpected finish reason")
    message = choice["message"]
    if message.get("tool_calls") or message.get("role") != "assistant":
        raise ValueError("Unexpected reply role or tool call")
    content = message["content"]
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("Expected a text assistant completion")
    raw_usage = reply.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        usage = {key: raw_usage[key] for key in ("prompt_tokens", "completion_tokens")
                 if type(raw_usage.get(key)) is int and raw_usage[key] >= 0}
    return content, choice["finish_reason"], usage


def complete(port, example, alias):
    connection = HTTPConnection("127.0.0.1", port, timeout=REQUEST_SECONDS)
    try:
        connection.request("POST", "/v1/chat/completions", json_bytes(request_body(example, alias)),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"Chat endpoint returned HTTP {response.status}")
        raw = response.read(2_097_153)
        if len(raw) > 2_097_152:
            raise ValueError("Response size exceeds cap")
        return parse_completion(raw, alias)
    finally:
        connection.close()


class CaseTimeout(Exception):
    pass


def evaluate_case(port, example, alias, deadline):
    started = time.monotonic()
    window = min(REQUEST_SECONDS, deadline - started)
    if window <= 0:
        return None, None
    signal.setitimer(signal.ITIMER_REAL, window)
    try:
        try:
            answer, finish, usage = complete(port, example, alias)
        except (CaseTimeout, OSError, HTTPException, ValueError, TypeError, KeyError) as error:
            result = {"status": "request_error", "kind": type(error).__name__}
            answer = None
        else:
            result = {"finish_reason": finish, **usage,
                      "status": "model_cap" if finish == "length" else "scored"}
            try:
                result.update(score(example, answer))
            except Exception as error:
                result.update(status="scoring_error", kind=type(error).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    return result, answer


def save_receipt(path, result):
    temp = path.with_suffix(".new")
    temp.write_bytes(json_bytes(result))
    temp.replace(path)


def run(examples, frozen, args):
    admission = validate_admission(args.admission_fd, args.lock_fd, args.stage, args.arm)
    port = local_port(admission["endpoint"])
    deadline = time.monotonic() + admission["wall_seconds"]
    output = Path(args.output).resolve()
    if output.parent != ROOT:
        raise ValueError("Choose a new private output directory directly under the preparation directory")
    output.mkdir(mode=0o700)
    receipt = output / "arm.json"
    result = {"schema": 1, "stage": args.stage, "arm": args.arm, "status": "interrupted",
              **{name: admission[name] for name in SHARED}, "samples": {}}
    save_receipt(receipt, result)

    def alarm(_signum, _frame):
        raise CaseTimeout()

    signal.signal(signal.SIGALRM, alarm)
    with (output / "private-responses.jsonl").open("x") as transcript:
        for key in frozen["stages"][args.stage]:
            outcome, answer = evaluate_case(port, examples[key], admission["model"], deadline)
            if outcome is None:
                break
            if answer is not None:
                transcript.write(json.dumps({"id": key, "response": answer}) + "\n")
                transcript.flush()
            result["samples"][str(key)] = outcome
            save_receipt(receipt, result)
            if outcome["status"] == "request_error":
                break  # Endpoint/model/transport errors require owner diagnosis before more requests.
        else:
            result["status"] = "finished"
            save_receipt(receipt, result)


def validate_receipt(receipt, frozen, examples, arm):
    stage = receipt.get("stage")
    if receipt.get("schema") != 1 or receipt.get("arm") != arm or stage not in WALL_LIMITS:
        raise ValueError("Unexpected receipt schema, stage or arm")
    if receipt.get("status") not in {"finished", "interrupted"}:
        raise ValueError("Unexpected arm status")
    if receipt.get("freeze_sha256") != digest(ROOT / "freeze.json") or receipt.get("program_sha256") != digest(__file__):
        raise ValueError("Arm receipt differs from program or freeze")
    if any(not isinstance(receipt.get(name), str) or not SHA.fullmatch(receipt[name]) for name in SHARED[1:]):
        raise ValueError("Arm receipt is missing an identity hash")
    if not isinstance(receipt.get("model"), str) or not MODEL.fullmatch(receipt["model"]):
        raise ValueError("Arm receipt is missing a local alias")
    selected = {str(key) for key in frozen["stages"][stage]}
    samples = receipt["samples"]
    if not samples.keys() <= selected:
        raise ValueError("Unknown ID in arm receipt")
    for key, entry in samples.items():
        status = entry.get("status")
        if status in {"scored", "model_cap"}:
            count = len(examples[int(key)].instruction_id_list)
            for mode in ("strict", "loose"):
                values = entry.get(mode)
                if not isinstance(values, list) or len(values) != count or any(type(value) is not bool for value in values):
                    raise ValueError("Invalid count or value for scored instructions")
            if any(a and not b for a, b in zip(entry["strict"], entry["loose"], strict=True)):
                raise ValueError("A strict pass must also pass loose scoring")
        elif status not in {"request_error", "scoring_error"}:
            raise ValueError("Unknown status in arm receipt")
    if receipt["status"] == "finished" and samples.keys() != selected:
        raise ValueError("Finished arm lacks planned IDs")


def empty_tally():
    return {key: 0 for key in (*OUTCOMES, "off_pass", "off_missing", "on_pass", "on_missing")}


def add_counts(tally, off, on):
    for key, value in (("off", off), ("on", on)):
        tally[f"{key}_pass"] += value is True
        tally[f"{key}_missing"] += value is None
    if off is None or on is None:
        paired = "missing"
    elif off and on:
        paired = "both_pass"
    elif not off and not on:
        paired = "both_fail"
    else:
        paired = "off_only" if off else "on_only"
    tally[paired] += 1


def with_rates(tally):
    denominator = sum(tally[key] for key in OUTCOMES)
    off_resolved = denominator - tally["off_missing"]
    on_resolved = denominator - tally["on_missing"]
    return {
        **tally, "planned": denominator,
        "off_resolved": off_resolved, "on_resolved": on_resolved,
        "off_accuracy": tally["off_pass"] / denominator if off_resolved == denominator else None,
        "on_accuracy": tally["on_pass"] / denominator if on_resolved == denominator else None,
        "on_minus_off": ((tally["on_only"] - tally["off_only"]) / denominator
                         if not tally["missing"] else None),
    }


def paired_report(examples, frozen, off, on):
    validate_receipt(off, frozen, examples, "off")
    validate_receipt(on, frozen, examples, "on")
    if off["stage"] != on["stage"] or any(off[name] != on[name] for name in SHARED):
        raise ValueError("Arm identities or stages do not match")
    modes = {mode: {"prompt": empty_tally(), "instruction": empty_tally(), "by_instruction": {}}
             for mode in ("strict", "loose")}
    statuses = {arm: dict.fromkeys(("scored", "model_cap", "request_error", "scoring_error", "absent"), 0)
                for arm in ("off", "on")}
    missing_pairs = []
    for key in frozen["stages"][off["stage"]]:
        entries = {arm: receipt["samples"].get(str(key), {"status": "absent"})
                   for arm, receipt in (("off", off), ("on", on))}
        for arm, entry in entries.items():
            statuses[arm][entry["status"]] += 1
        if any(entry["status"] not in {"scored", "model_cap"} for entry in entries.values()):
            missing_pairs.append(key)
        ids = examples[key].instruction_id_list
        for mode, tally in modes.items():
            a = entries["off"].get(mode, [None] * len(ids))
            b = entries["on"].get(mode, [None] * len(ids))
            add_counts(tally["prompt"], None if a[0] is None else all(a),
                       None if b[0] is None else all(b))
            for name, av, bv in zip(ids, a, b, strict=True):
                add_counts(tally["instruction"], av, bv)
                add_counts(tally["by_instruction"].setdefault(name, empty_tally()), av, bv)
    for tally in modes.values():
        tally["prompt"] = with_rates(tally["prompt"])
        tally["instruction"] = with_rates(tally["instruction"])
        tally["by_instruction"] = {name: with_rates(counts) for name, counts in tally["by_instruction"].items()}
    return {
        "dataset": frozen["dataset"], "revision": frozen["revision"], "stage": off["stage"],
        "identities": {name: off[name] for name in SHARED},
        "arm_run_status": {arm: receipt["status"] for arm, receipt in (("off", off), ("on", on))},
        "arm_case_counts": statuses, "missing_pair_ids": missing_pairs, "metrics": modes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    running = commands.add_parser("run")
    running.add_argument("--stage", choices=WALL_LIMITS, required=True)
    running.add_argument("--arm", choices=("off", "on"), required=True)
    running.add_argument("--admission-fd", required=True, type=int)
    running.add_argument("--lock-fd", required=True, type=int)
    running.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("off_receipt")
    report.add_argument("on_receipt")
    args = parser.parse_args()
    examples = load_examples()
    frozen = verify_manifest(examples)
    if args.command == "verify":
        setup_scorer()
        print(json_bytes({"stages": {stage: len(ids) for stage, ids in frozen["stages"].items()},
                          "instruction_counts": frozen["instruction_counts"],
                          "smoke_ids": frozen["stages"]["smoke"]}).decode(), end="")
    elif args.command == "run":
        setup_scorer()
        run(examples, frozen, args)
    else:
        print(json_bytes(paired_report(examples, frozen, read_json(args.off_receipt),
                                     read_json(args.on_receipt))).decode(), end="")


if __name__ == "__main__":
    os.umask(0o077)
    main()
