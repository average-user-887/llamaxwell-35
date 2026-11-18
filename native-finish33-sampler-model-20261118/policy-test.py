#!/usr/bin/env python3
"""Exercise the exact compiled opt-in and rewind predicate without model or GPU."""
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
server = (HERE / "server-context.cpp").read_text()
common = (HERE.parent / "night-cache-only/src/common/common.h").read_text()
modes = re.search(r"enum common_context_seq_rm_type \{.*?\n\};", common, re.S).group(0)
start = server.index("static bool sampler_snapshot_elision_enabled()")
end = server.index("// Accessed only by the server worker.", start)
contents = """#include <cstdint>
#include <cstdlib>
#include <cstring>
""" + modes + "\n" + server[start:end] + """
int main(int argc, char ** argv) {
    if (argc != 2 || sampler_snapshot_elision_enabled() != (std::strcmp(argv[1], "1") == 0)) {
        return 1;
    }
    const auto modes = {
        COMMON_CONTEXT_SEQ_RM_TYPE_NO, COMMON_CONTEXT_SEQ_RM_TYPE_PART,
        COMMON_CONTEXT_SEQ_RM_TYPE_FULL, COMMON_CONTEXT_SEQ_RM_TYPE_RS,
    };
    for (bool enabled : {false, true}) {
        for (bool replay : {false, true}) {
            for (auto mode : modes) {
                for (size_t draft = 0; draft < 5; ++draft) {
                    for (uint32_t retained = 0; retained < 5; ++retained) {
                        const bool skip = may_elide_sampler_snapshot(enabled, replay, mode, draft, retained);
                        const bool must_keep = !enabled || replay || mode != COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                                               draft == 0 || retained < draft;
                        if (skip == must_keep) return 2;
                        for (size_t rejected = 0; rejected <= draft; ++rejected) {
                            const bool restores = rejected > 0 && (mode == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                (mode == COMMON_CONTEXT_SEQ_RM_TYPE_RS && rejected > retained));
                            if (skip && restores) return 3;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
"""
with TemporaryDirectory(dir=HERE) as tmp:
    source = Path(tmp) / "policy.cpp"
    executable = Path(tmp) / "policy"
    source.write_text("#include <initializer_list>\n" + contents)
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", source, "-o", executable], check=True)
    var = "LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE"
    for value in (None, "", "0", "1", "true", "01"):
        env = os.environ.copy()
        env.pop(var, None)
        if value is not None:
            env[var] = value
        subprocess.run([executable, str(int(value == "1"))], env=env, check=True)
print("Compiled policy passed default-off, exact opt-in, bounded rollback and conservative mode/replay controls.")
