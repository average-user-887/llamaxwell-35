#!/usr/bin/env -S python3 -B
"""Frozen paired BFCL protocol. Running a model requires a supervisor admission."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib import metadata
import json
from math import comb
import os
from pathlib import Path
import re
import signal
import stat
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
VENV = Path.home() / ".local/share/lab-evals/venv"
SITE = VENV / "lib/python3.12/site-packages"
DATA = Path.home() / ".cache/inspect_evals/BFCL"
BFCL = SITE / "inspect_evals/bfcl"
LOCK = ROOT.parents[1] / "gpu-bench.lock"
SALT = "llamaxwell-35-bfcl-mtp-pairs-v1"
GROUPS = ("multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param")
SIMULATORS = (
    "gorilla_file_system", "long_context", "math_api", "message_api", "posting_api",
    "ticket_api", "trading_bot", "travel_booking", "vehicle_control",
)
OPTIONS = {
    "seed": 7, "temperature": 0.0, "max_tokens": 512,
    "parallel_tool_calls": True, "internal_tools": False, "reasoning_summary": "none",
    "extra_body": {"cache_prompt": False, "temperature": 0.0}, "max_retries": 0, "timeout": 90,
    "attempt_timeout": 90, "max_connections": 1,
}
LIMITS = {"message_limit": 128, "turn_limit": 32, "token_limit": 131072, "time_limit": 360}
SIZES = {"smoke": 4, "early": 16, "preview": 40, "full": 160}
STAGE_SECONDS = {"smoke": 1500, "early": 3600, "preview": 9000, "full": 36000}
ALPHA = 0.05
MARGIN = 0.05
IDENTITY = ("binary_sha256", "weights_sha256", "template_sha256", "profile_pair_sha256")
PACKAGES = ("inspect_ai", "inspect_evals", "openai", "httpx", "httpcore", "mpmath", "pydantic")
HEX = re.compile(r"[0-9a-f]{64}\Z")
MODEL = re.compile(r"(?:qwen|llamaxwell)[a-zA-Z0-9._-]{0,92}\Z", re.IGNORECASE)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        from hashlib import file_digest
        return file_digest(handle, "sha256").hexdigest()


def hash_rank(label, value):
    return sha256(f"{SALT}\0{label}\0{value}".encode()).hexdigest()


def source_paths():
    paths = {"bfcl_data/BFCL_v4_simple_python.json": DATA / "BFCL_v4_simple_python.json"}
    for category in GROUPS:
        name = f"BFCL_v4_{category}.json"
        for relative in (name, f"possible_answer/{name}"):
            paths[f"bfcl_data/{relative}"] = DATA / relative
    for module in SIMULATORS:
        relative = f"func_source_code/{module}.py"
        paths[f"bfcl_data/{relative}"] = DATA / relative
    for path in sorted((DATA / "multi_turn_func_doc").glob("*.json")):
        paths[f"bfcl_data/multi_turn_func_doc/{path.name}"] = path
    for directory in (BFCL, SITE / "inspect_ai", SITE / "inspect_evals/metadata", SITE / "inspect_evals/utils"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".jinja", ".j2"}:
                paths[f"venv/{path.relative_to(SITE)}"] = path
    for package in PACKAGES:
        dist = metadata.distribution(package)
        for record in dist.files or ():
            if record.name in {"METADATA", "RECORD", "direct_url.json"} and ".dist-info/" in str(record):
                paths[f"venv/{record}"] = dist.locate_file(record)
    return paths


def jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidates():
    all_rows = {}
    for category in GROUPS:
        source = f"BFCL_v4_{category}.json"
        questions = jsonl(DATA / source)
        answers = jsonl(DATA / "possible_answer" / source)
        ids = {row["id"] for row in questions}
        if len(questions) != 200 or len(ids) != 200 or len(answers) != 200 or ids != {row["id"] for row in answers}:
            raise ValueError(f"Unexpected question or answer population for {category}")
        eligible = []
        for row in questions:
            match = re.fullmatch(category + r"_(\d+)", row["id"])
            if match is None:
                raise ValueError("Unexpected public ID format")
            user_turns = sum(
                any(msg.get("role", "user") == "user" and str(msg.get("content", "")).strip() for msg in turn)
                for turn in row["question"]
            )
            if user_turns >= 2:
                eligible.append({"id": row["id"], "category": category,
                                 "family": int(match[1]), "user_turns": user_turns})
        all_rows[category] = eligible
    return all_rows


def select(rows):
    used = set()

    def choose(label, category):
        ordered = sorted(rows[category], key=lambda row: (hash_rank(label, row["id"]), row["id"]))
        result = next(row for row in ordered if row["family"] not in used)
        used.add(result["family"])
        return result

    smoke = [choose("smoke", category) for category in GROUPS]
    # Round-robin makes every stage prefix balanced and prevents late-stratum selection bias.
    cohort = [choose("cohort", category) for _ in range(40) for category in GROUPS]
    return smoke, cohort


def build_manifest():
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise ValueError("Use the existing lab-evals venv")
    rows = candidates()
    smoke, cohort = select(rows)
    sources = source_paths()
    return {
        "schema": 1, "salt": SALT, "scorer": "Inspect BFCL 6-B",
        "official_gorilla_commit": "dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a",
        "inspect_evals_commit": "32b79a244f17ecf54fa1186d30f33dc10143c10f",
        "python": sys.version.split()[0], "groups": list(GROUPS),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "population": {group: len(rows[group]) for group in GROUPS},
        "smoke": smoke, "cohort": cohort, "options": OPTIONS, "limits": LIMITS,
        "stage_sizes": SIZES, "stage_wall_seconds_per_arm_max": STAGE_SECONDS,
        "margin": MARGIN, "one_sided_alpha": ALPHA,
        "files": {name: digest(path) for name, path in sources.items()},
    }


def manifest():
    frozen = read(ROOT / "freeze.json")
    if frozen != build_manifest():
        raise ValueError("Current source, evaluator, or parameters differ from the freeze")
    return frozen


def stage_rows(frozen, stage):
    return frozen["smoke"] if stage == "smoke" else frozen["cohort"][:SIZES[stage]]


def endpoint_parts(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/v1":
        raise ValueError("An explicit http://127.0.0.1:PORT/v1 endpoint is required")
    if not parsed.port or parsed.netloc != f"127.0.0.1:{parsed.port}" or parsed.query or parsed.fragment:
        raise ValueError("Endpoint credentials, default ports and URL suffixes are unsupported")
    if url != f"http://127.0.0.1:{parsed.port}/v1":
        raise ValueError("Endpoint must be canonical")
    return parsed.port


def validate_admission(fd, lock_fd, stage, arm):
    with os.fdopen(os.dup(fd)) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Admission must be furnished through a private supervisor-owned file descriptor")
        raw = handle.read(16385)
    if len(raw) > 16384:
        raise ValueError("Admission record is too large")
    data = json.loads(raw)
    if data.get("arm") != arm or data.get("stage") != stage or data.get("r2_same_build") is not True:
        raise ValueError("Admission arm, stage or same-build attestation mismatches")
    if data.get("freeze_sha256") != digest(ROOT / "freeze.json") or data.get("program_sha256") != digest(__file__):
        raise ValueError("Admission does not bind this program and freeze")
    if any(not isinstance(data.get(key), str) or not HEX.fullmatch(data[key]) for key in IDENTITY):
        raise ValueError("Admission must provide every shared identity hash")
    if not isinstance(data.get("model"), str) or not MODEL.fullmatch(data["model"]):
        raise ValueError("Admission must pin the same local Qwen/LlamaXwell alias")
    wall = data.get("wall_seconds")
    if type(wall) is not int or not 1 <= wall <= STAGE_SECONDS[stage]:
        raise ValueError("Admission wall budget is missing or exceeds frozen stage cap")
    endpoint_parts(data["endpoint"])
    opened = os.fstat(lock_fd)
    expected = LOCK.stat()
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Supervisor did not pass the campaign GPU lock")
    lock_info = Path(f"/proc/self/fdinfo/{lock_fd}").read_text()
    if not any("FLOCK" in line and "WRITE" in line for line in lock_info.splitlines() if line.startswith("lock:")):
        raise ValueError("Supervisor GPU lock descriptor has no exclusive flock")
    return data


def dataset_task(rows):
    from inspect_ai.dataset import MemoryDataset
    from inspect_evals.bfcl.backends import loader
    from inspect_evals.bfcl.bfcl import DATASET_LOCATION, FUNC_SOURCE_LOCATION, bfcl

    if DATASET_LOCATION.resolve() != DATA.resolve() or FUNC_SOURCE_LOCATION.resolve() != (DATA / "func_source_code").resolve():
        raise ValueError("Inspect selected a different BFCL cache")
    if not (DATA / "BFCL_v4_simple_python.json").is_file() or not FUNC_SOURCE_LOCATION.is_dir():
        raise ValueError("Pinned BFCL cache canaries are absent")
    task = bfcl(categories=list(GROUPS))
    if loader._state["load_errors"] or len(loader._state["loaded_classes"]) != 8:
        raise RuntimeError("Official simulator inventory is incomplete")
    available = {sample.id: sample for sample in task.dataset}
    ordered = [available[row["id"]] for row in rows]
    if any(sample.metadata["category_name"] != row["category"] for sample, row in zip(ordered, rows, strict=True)):
        raise RuntimeError("Inspect category disagrees with frozen selection")
    task.dataset = MemoryDataset(ordered, name="Frozen paired BFCL", location=str(DATA))
    return task


def model_for(endpoint, alias, transport):
    import httpx
    from inspect_ai.model import get_model

    endpoint_parts(endpoint)
    client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                               limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
    return get_model(f"openai/{alias}", base_url=endpoint, api_key="local-no-credential",
                     responses_api=False, http_client=client, max_retries=0)


def request_guard(request, endpoint, alias):
    if request.method != "POST" or str(request.url) != f"{endpoint}/chat/completions":
        raise RuntimeError("Only the admitted loopback chat-completions route is permitted")
    if request.headers.get("Authorization") != "Bearer local-no-credential":
        raise RuntimeError("The local fixture header is required; real credentials cannot be sent")
    if request.headers.get("OpenAI-Organization") or request.headers.get("OpenAI-Project"):
        raise RuntimeError("Cloud account headers cannot be sent")
    body = json.loads(request.content)
    checks = {key: OPTIONS[key] for key in ("seed", "temperature")}
    checks["cache_prompt"] = OPTIONS["extra_body"]["cache_prompt"]
    checks["model"] = alias
    cap = body.get("max_completion_tokens", body.get("max_tokens"))
    if cap != OPTIONS["max_tokens"] or any(body.get(key) != value for key, value in checks.items()):
        raise RuntimeError("Outbound model request differs from frozen generation options")
    if body.get("tools") and body.get("parallel_tool_calls") is not True:
        raise RuntimeError("Outbound tool request differs from frozen parallel-call option")


def transport_for(endpoint, alias):
    import httpx

    class AdmittedTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            request_guard(request, endpoint, alias)
            return await super().handle_async_request(request)

    return AdmittedTransport(retries=0, trust_env=False)


def evaluate(task, model, log_dir):
    from inspect_ai import eval as inspect_eval

    # Inspect probes Git for unregistered tasks. The program and evaluator are
    # already hash-pinned, and this worker must run without executable discovery.
    overrides = {
        "PATH": str(ROOT / ".no-executables"),
        "XDG_DATA_HOME": str(log_dir / "xdg-data"),
        "XDG_CACHE_HOME": str(log_dir / "xdg-cache"),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        return inspect_eval(task, model=model, log_dir=str(log_dir), display="none",
                            score_display=False, max_samples=1, max_tasks=1, epochs=1,
                            sample_shuffle=False, fail_on_error=False, retry_on_error=0,
                            score_on_error=False, log_samples=True, log_images=False,
                            log_model_api=False, log_level="warning", **LIMITS, **OPTIONS)[0]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def outcome(sample):
    limit = sample.limit.type if sample.limit is not None else None
    if limit in {"token", "message", "turn"}:
        return {"status": "model_cap", "score": 0, "limit": limit}
    if sample.error is not None or limit is not None:
        return {"status": "infra", "score": None}
    scores = list((sample.scores or {}).values())
    if len(scores) != 1 or type(scores[0].value) not in {int, float} or scores[0].value not in {0, 1}:
        return {"status": "infra", "score": None}
    return {"status": "scored", "score": int(scores[0].value)}


def run(stage, arm, admission_fd, lock_fd, output):
    admission = validate_admission(admission_fd, lock_fd, stage, arm)
    frozen = manifest()
    rows = stage_rows(frozen, stage)
    endpoint = admission["endpoint"]
    port = endpoint_parts(endpoint)

    def guard(event, args):
        if event in {"subprocess.Popen", "os.system", "socket.bind"}:
            raise PermissionError("External executors and listeners are disabled")
        if event == "socket.connect" and args[1] != ("127.0.0.1", port):
            raise PermissionError("Only the admitted loopback socket is permitted")
        if event == "socket.getaddrinfo" and args[0] != "127.0.0.1":
            raise PermissionError("DNS and external sockets are disabled")

    os.umask(0o077)
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Run output must stay within the isolated preparation directory")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for key in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_SAFETY_IDENTIFIER"):
        os.environ.pop(key, None)
    sys.addaudithook(guard)
    # Load benchmark data and permitted mock functions before creating a model client.
    task = dataset_task(rows)
    model = model_for(endpoint, admission["model"], transport_for(endpoint, admission["model"]))
    result = {
        "schema": 1, "stage": stage, "arm": arm, "model": admission["model"],
        "freeze_sha256": admission["freeze_sha256"], "program_sha256": admission["program_sha256"],
        **{key: admission[key] for key in IDENTITY},
        "inspect_status": "interrupted", "samples": {},
    }
    receipt = output / "arm.json"
    receipt.write_bytes(json_bytes(result))

    def deadline(_signum, _frame):
        # Do not let a per-sample error handler catch an expired stage deadline.
        os._exit(124)

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(admission["wall_seconds"])
    try:
        log = evaluate(task, model, output / "private")
    finally:
        signal.alarm(0)
    samples = {}
    selected_ids = {row["id"] for row in rows}
    for sample in log.samples or ():
        if sample.id in samples or sample.id not in selected_ids:
            raise RuntimeError("Unexpected or duplicate Inspect sample ID")
        samples[sample.id] = outcome(sample)
    result.update(inspect_status=log.status, samples=samples)
    temporary_receipt = output / "arm.json.new"
    temporary_receipt.write_bytes(json_bytes(result))
    temporary_receipt.replace(receipt)


def binomial_cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))


def upper_loss_bound(losses, n, alpha=ALPHA):
    # Above mean threshold n*p >= k+1, the binomial low tail dominates the
    # Poisson-binomial low tail (Hoeffding, 1956). This allows unequal strata.
    if losses >= n:
        return 1.0
    left, right = (losses + 1) / n, 1.0
    for _ in range(60):
        middle = (left + right) / 2
        if binomial_cdf(losses, n, middle) > alpha:
            left = middle
        else:
            right = middle
    return right


def paired_report(frozen, baseline, candidate):
    stage = baseline["stage"]
    bind = ("schema", "stage", "model", "freeze_sha256", "program_sha256", *IDENTITY)
    if baseline["arm"] != "off" or candidate["arm"] != "on" or any(baseline[key] != candidate[key] for key in bind):
        raise ValueError("Arm receipts do not match stage, model, freeze or same-build identities")
    rows = stage_rows(frozen, stage)
    allowed = {row["id"] for row in rows}
    names = ("both_pass", "both_fail", "on_only", "off_only", "off_infra", "on_infra",
             "off_missing", "on_missing", "off_model_cap", "on_model_cap", "unresolved_pairs")
    counts = {group: Counter({name: 0 for name in names}) for group in GROUPS}
    for arm in (baseline, candidate):
        if set(arm["samples"]) - allowed:
            raise ValueError("Receipt contains samples outside its frozen stage")
    for row in rows:
        count = counts[row["category"]]
        samples = [arm["samples"].get(row["id"]) for arm in (baseline, candidate)]
        valid = True
        for label, sample in zip(("off", "on"), samples, strict=True):
            if sample is None:
                count[f"{label}_missing"] += 1
                valid = False
            elif sample["status"] == "infra" and sample["score"] is None:
                count[f"{label}_infra"] += 1
                valid = False
            elif sample["status"] == "model_cap" and sample["score"] == 0:
                count[f"{label}_model_cap"] += 1
            elif sample["status"] != "scored" or type(sample["score"]) is not int or sample["score"] not in {0, 1}:
                raise ValueError("Invalid arm sample record")
        if valid:
            scores = tuple(sample["score"] for sample in samples)
            count[{(1, 1): "both_pass", (0, 0): "both_fail", (0, 1): "on_only", (1, 0): "off_only"}[scores]] += 1
        else:
            count["unresolved_pairs"] += 1
    aggregate = dict(sum(counts.values(), Counter()))
    total = len(rows)
    unresolved = aggregate.get("unresolved_pairs", 0)
    losses = aggregate.get("off_only", 0)
    gains = aggregate.get("on_only", 0)
    lower = -upper_loss_bound(losses + unresolved, total)
    both = aggregate.get("both_pass", 0)
    delta = (gains - losses) / total if not unresolved else None
    return {
        "stage": stage, "planned_pairs": total, "resolved_pairs": total - unresolved,
        "groups": {name: dict(count) for name, count in counts.items()}, "total": aggregate,
        "off_accuracy": (both + losses) / total if not unresolved else None,
        "on_accuracy": (both + gains) / total if not unresolved else None,
        "paired_delta": delta, "paired_delta_lower_95": lower,
        "noninferiority": (
            "eligible_pass" if stage == "full" and not unresolved and lower > -MARGIN
            and baseline["inspect_status"] == candidate["inspect_status"] == "success"
            else "unresolved"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("verify")
    report = command.add_parser("report")
    report.add_argument("off", type=Path)
    report.add_argument("on", type=Path)
    launch = command.add_parser("run")
    launch.add_argument("--stage", choices=SIZES, required=True)
    launch.add_argument("--arm", choices=("off", "on"), required=True)
    launch.add_argument("--admission-fd", type=int, required=True)
    launch.add_argument("--lock-fd", type=int, required=True)
    launch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        frozen = manifest()
        rows = frozen["smoke"] + frozen["cohort"]
        if len({row["family"] for row in rows}) != 164 or any(row["user_turns"] < 2 for row in rows):
            raise ValueError("Invalid family or user-turn allocation")
        dataset_task(rows)
        print("Verified 160 cohort IDs, four disjoint smoke IDs, all source hashes and all eight simulators.")
    elif args.command == "report":
        frozen = manifest()
        baseline, candidate = read(args.off), read(args.on)
        if any(arm["freeze_sha256"] != digest(ROOT / "freeze.json") or arm["program_sha256"] != digest(__file__)
               for arm in (baseline, candidate)):
            raise ValueError("Receipts are not bound to this documented program and freeze")
        print(json_bytes(paired_report(frozen, baseline, candidate)).decode(), end="")
    else:
        run(args.stage, args.arm, args.admission_fd, args.lock_fd, args.output)


if __name__ == "__main__":
    main()
