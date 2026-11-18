#!/usr/bin/env python3
"""Generate a server-only sampler snapshot change from the pinned N1 source."""
import difflib
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "night-cache-only/src/tools/server/server-context.cpp"
SOURCE_SHA = "1e20f71d8f202245db9e347a3aa97390890b41f0ec32f2de5d8a91ffb18be5da"


def main():
    original = BASE.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError("pinned N1 server-context.cpp bytes differ")
    source = original.decode()

    def replace(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError(f"server source anchor changed: {before}")
        source = source.replace(before, after)

    replace("static mv1_host_t g_mv1;\n", """static mv1_host_t g_mv1;

static bool sampler_snapshot_elision_enabled() {
    static const bool enabled = [] {
        const char * flag = getenv("LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE");
        return flag != nullptr && strcmp(flag, "1") == 0;
    }();
    return enabled;
}

""" + (HERE / "snapshot-policy.inc").read_text() + """
// Accessed only by the server worker. Log snapshots at completion boundaries.
static struct {
    unsigned long long elided = 0, elided_rejected = 0, kept = 0, restored = 0;
} g_sampler_snapshot;
""")
    replace("    void send_final_response(server_slot & slot) {\n", """    void send_final_response(server_slot & slot) {
        if (sampler_snapshot_elision_enabled()) {
            fprintf(stderr, "sampler_snapshot: elided=%llu elided_rejected=%llu kept=%llu restored=%llu\\n",
                    g_sampler_snapshot.elided, g_sampler_snapshot.elided_rejected,
                    g_sampler_snapshot.kept, g_sampler_snapshot.restored);
        }
""")
    replace("                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));",
            """                const bool snapshot_opt_in = sampler_snapshot_elision_enabled();
                const bool elide_snapshot = snapshot_opt_in && may_elide_sampler_snapshot(
                    snapshot_opt_in, slot.spec_is_replay, ctx_tgt_seq_rm_type, n_draft, llama_n_rs_seq(ctx_tgt));
                common_sampler_ptr smpl_save;
                if (elide_snapshot) {
                    ++g_sampler_snapshot.elided;
                } else {
                    smpl_save.reset(common_sampler_clone(slot.smpl.get()));
                    if (snapshot_opt_in) {
                        ++g_sampler_snapshot.kept;
                    }
                }""")
    replace("                // check for partial draft acceptance\n", """                if (elide_snapshot && n_rollback > 0) {
                    ++g_sampler_snapshot.elided_rejected;
                }

                // check for partial draft acceptance
""")
    replace("                        common_sampler_copy(smpl_save.get(), slot.smpl.get());",
            """                        GGML_ASSERT(smpl_save != nullptr);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());
                        if (snapshot_opt_in) {
                            ++g_sampler_snapshot.restored;
                        }""")

    (HERE / "server-context.cpp").write_text(source)
    patch = difflib.unified_diff(original.decode().splitlines(keepends=True), source.splitlines(keepends=True),
                                 fromfile="a/tools/server/server-context.cpp", tofile="b/tools/server/server-context.cpp")
    (HERE / "server-context.patch").write_text("".join(patch))


if __name__ == "__main__":
    main()
