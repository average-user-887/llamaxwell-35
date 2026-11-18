#!/usr/bin/env python3
"""Stage an isolated N40 draft overlay after a static ABI and parent check."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
C = HERE.parent
R1 = C / "finish33-sampler-r1"
N1 = C / "night-cache-only"
N40 = C / "night-one-graph-draft"
NAME = ("libllama.so.0.6.0", "libllama-common.so.0.6.0")
SERVER = "libllama-server-impl.so"
META_TAGS = ("NEEDED", "SONAME", "RUNPATH", "RPATH")


def output(*args):
    return subprocess.check_output(args, text=True)


def sha(path):
    return output("sha256sum", path).split()[0]


def symbols(path, kind):
    lines = output("nm", "-D", f"--{kind}-only", path).splitlines()
    return {tuple(line.split()[-2:]) for line in lines if line.split()}


def loader_fields(path):
    return [line.split(")", 1)[1].strip() for line in output("readelf", "-d", path).splitlines()
            if any(f"({tag})" in line for tag in META_TAGS)]


def finish33(baseline):
    spec = importlib.util.spec_from_file_location("hybrid_stage_finish33", R1 / "validate.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = verifier.verify("native", baseline)
    assert result["raw_runner_exit"] == result["raw_cleanup_exit"] == 0
    assert baseline.parent == R1 / "receipts"
    return result["validated"]["actual_server_software_sha256"]


def compat(software):
    for manifest in ("FROZEN.sha256", "SOURCE.sha256"):
        subprocess.run(["sha256sum", "--check", "--strict", "--quiet", manifest], cwd=N40, check=True)
    assert "exit_code=0" in (N40 / "BUILD_STATUS").read_text().splitlines()
    assert sha(R1 / SERVER) == software[SERVER]
    assert sha(C / "finish33-r2-stride/libggml-cuda.so") == software["libggml-cuda.so"]
    package = json.loads((N40 / "OBJECTS.json").read_text())
    old_digests, new_digests, abi = {}, {}, {}
    new_core_exports = symbols(N40 / "libs" / NAME[0], "defined")
    for name in NAME:
        parent, candidate = N1 / "libs" / name, N40 / "libs" / name
        before, after = symbols(parent, "defined"), symbols(candidate, "defined")
        assert loader_fields(parent) == loader_fields(candidate) and before <= after
        new_imports = symbols(candidate, "undefined") - symbols(parent, "undefined")
        assert {symbol for _, symbol in new_imports} <= {symbol for _, symbol in new_core_exports}
        old_digests[name] = sha(parent)
        new_digests[name] = sha(candidate)
        assert old_digests[name] == software[name]
        assert new_digests[name] == package[name] == sha(N40 / "build/bin" / name)
        assert new_digests[name] != old_digests[name]
        abi[name] = {"all_original_dynamic_definitions_preserved": True,
                     "equal_dependency_soname_search_path_metadata": True,
                     "new_definitions": sorted(symbol for _, symbol in after - before),
                     "new_imports_satisfied_by_alternate_core": sorted(symbol for _, symbol in new_imports)}
    server_imports = symbols(R1 / SERVER, "undefined")
    before = set().union(*(symbols(N1 / "libs" / name, "defined") for name in NAME))
    after = set().union(*(symbols(N40 / "libs" / name, "defined") for name in NAME))
    owners = {symbol for _, symbol in server_imports} & {symbol for _, symbol in before}
    assert owners <= {symbol for _, symbol in after}
    assert any("libllama-common.so.0" in text for text in loader_fields(R1 / SERVER))
    return {"original_libraries": old_digests, "alternate_libraries": new_digests, "abi": abi,
            "unchanged_server_imports_satisfied": len(owners),
            "n40_build_products_match_packaged_libraries": True, "n40_source_parent_hashes_verified": True}


def variant_release():
    text = (R1 / "release.sh").read_text()
    replacements = {
        'overlay=$campaign/finish33-r2-stride\n':
            'overlay=$campaign/finish33-r2-stride\nhybrid=$campaign/night-one-graph-draft/libs\n',
        '  "$campaign/last-ms-r2-selector-model/release.sh" --check\n  "$overlay/release.sh" --check\n':
            '  "$campaign/finish33-sampler-r1/release.sh" --check\n',
        '-v "$n1:$n1:ro" -v "$card/libllama-server-impl.so:$n1/libllama-server-impl.so:ro"':
            '-v "$n1:$n1:ro" -v "$hybrid/libllama.so.0.6.0:$n1/libllama.so.0.6.0:ro" '
            '-v "$hybrid/libllama-common.so.0.6.0:$n1/libllama-common.so.0.6.0:ro" '
            '-v "$card/libllama-server-impl.so:$n1/libllama-server-impl.so:ro"',
        '|LLAMA_TRACE|LLAMA_MV3_PHASE': '|LLAMA_TRACE|LLAMA_MTP_LAST_QUERY_TRACE|LLAMA_MV3_PHASE',
        'usage: release.sh --check | --run native|serial | --compare SERIAL_RECEIPT NATIVE_RECEIPT':
            'usage: release.sh --check | --run native | --compare TERMINAL_NATIVE_RECEIPT',
        '[[ $2 == serial || $2 == native ]]': '[[ $2 == native ]]',
        '--compare) [[ $# == 3 ]] || usage; check >&2; exec python3 -B "$probe" compare "$2" "$3" ;;':
            '--compare) [[ $# == 2 ]] || usage; check >&2; exec python3 -B "$probe" compare "$2" ;;',
        '--run) [[ $# == 2 && ( $2 == serial || $2 == native ) ]] || usage; check ;;':
            '--run) [[ $# == 2 && $2 == native ]] || usage; check ;;',
        'FINISH33_SAMPLER_RECEIPT=': 'HUNDRED_HYBRID_DRAFT_RECEIPT=',
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    assert text.count("nmmoon-finish33-sampler-") == 2
    return text.replace("nmmoon-finish33-sampler-", "nmmoon-hundred-hybrid-")


def pin(file, paths):
    (HERE / file).write_text("".join(f"{sha(path)}  {os.path.relpath(path, HERE)}\n" for path in paths))


def main():
    baseline = Path(sys.argv[1]).resolve()
    assert len(sys.argv) == 2 and not (HERE / "FROZEN.sha256").exists()
    software = finish33(baseline)
    binding = {"intent": "split versus merged draft re-evaluation with one final query",
               "mode": "native", "native_baseline": str(baseline.relative_to(C)),
               "baseline_mapped_software": software,
               "warm_historical_saving_ms_per_offer": 0.1605247524752471,
               "arm_cap_s": 430, "live_qualification": None, **compat(software)}
    (HERE / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    runner = HERE / "release.sh"
    runner.write_text(variant_release())
    runner.chmod(0o700)
    for name in ("supervise", "check-layout.py", SERVER):
        (HERE / name).symlink_to(Path("..") / R1.name / name)
    pin("INPUTS.sha256", [
        R1 / "FROZEN.sha256", R1 / "INPUTS.sha256", R1 / "validate.py", R1 / "release.sh",
        N40 / "FROZEN.sha256", N40 / "SOURCE.sha256", N40 / "OBJECTS.json", N40 / "last-query.patch",
        C / "finish33-r2-stride/libggml-cuda.so", baseline / "observations.sha256",
        *(N1 / "libs" / name for name in NAME), *(N40 / "libs" / name for name in NAME),
    ])
    subprocess.run(["bash", "-n", runner], check=True)
    subprocess.run(["python3", "-B", HERE / "model.py", "check"], check=True)
    pin("FROZEN.sha256", [HERE / name for name in (
        "stage.py", "model.py", "release.sh", "binding.json", "INPUTS.sha256", SERVER, "supervise", "check-layout.py")])
    print("Static parent, build products, ELF ABI and exact current split baseline validated. Live hybrid result unavailable.")


if __name__ == "__main__":
    main()
