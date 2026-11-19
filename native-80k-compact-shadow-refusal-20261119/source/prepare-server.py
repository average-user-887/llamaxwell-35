#!/usr/bin/env python3
from pathlib import Path
import difflib

here = Path(__file__).resolve().parent
source = here.parent/'finish33-sampler/server-context.cpp'


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


before = text = source.read_text()
text = change(text, '#include "sampling.h"\n', '#include "sampling.h"\n#include "compact-sampler-policy.h"\n')
text = change(text, 'static mv2_dump_t g_mv2;\n', '''static mv2_dump_t g_mv2;

enum class compact_mode { off, on, shadow };
static compact_mode compact_readback_mode() {
    static const auto mode = [] {
        const char * value = std::getenv("LLAMA_OCCAMY_COMPACT_READBACK");
        const char * scores = std::getenv("LLAMA_MV2_DUMP");
        if (!value || (scores && scores[0])) {
            return compact_mode::off;
        }
        if (std::string_view(value) == "1") {
            return compact_mode::on;
        }
        return std::string_view(value) == "shadow" ? compact_mode::shadow : compact_mode::off;
    }();
    return mode;
}

// The only writer is the server worker. Report cumulative byte counts per response.
static struct {
    unsigned long long requested = 0, launched = 0, compact_only = 0, ambiguous = 0, unsupported = 0;
    unsigned long long raw_bytes = 0, compact_bytes = 0, shadow_rows = 0, shadow_mismatches = 0;
} g_compact;

static void compact_record(const occamy_compact::Result & result) {
    ++g_compact.requested;
    g_compact.launched += result.launched;
    g_compact.compact_only += result.compact_only;
    g_compact.ambiguous += result.launched && !occamy_compact::valid(result.rows);
    g_compact.unsupported += !result.launched;
    g_compact.raw_bytes += result.raw;
    g_compact.compact_bytes += result.compact;
}
''')
text = change(text, '''    void send_final_response(server_slot & slot) {
        if (sampler_snapshot_elision_enabled()) {''', '''    void send_final_response(server_slot & slot) {
        if (compact_readback_mode() != compact_mode::off) {
            fprintf(stderr, "compact_readback: requested=%llu launched=%llu compact_only=%llu ambiguous=%llu unsupported=%llu raw_bytes=%llu compact_bytes=%llu shadow_rows=%llu shadow_mismatches=%llu\\n",
                    g_compact.requested, g_compact.launched, g_compact.compact_only, g_compact.ambiguous,
                    g_compact.unsupported, g_compact.raw_bytes, g_compact.compact_bytes,
                    g_compact.shadow_rows, g_compact.shadow_mismatches);
        }
        if (sampler_snapshot_elision_enabled()) {''')
text = change(text, '''            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {''', '''            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            occamy_compact::Result compact;
            try {''')
text = change(text, '                bool ok = decode(n_batch, off);', '                bool ok = decode(n_batch, off, compact);')
text = change(text, '                post_decode(n_tokens, off);', '                post_decode(n_tokens, off, compact);')
text = change(text, '''    bool decode(int32_t & n_batch, int32_t off) {''', '''    bool compact_eligible(int32_t off, occamy_compact::Mask & mask) const {
        if (compact_readback_mode() == compact_mode::off || !spec || slots.size() != 1 ||
                params_base.speculative.types.size() != 1 || params_base.speculative.types.front() != COMMON_SPECULATIVE_TYPE_DRAFT_MTP ||
                ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS || llama_n_rs_seq(ctx_tgt) < 1 ||
                off != 0 || batch.size() != 2 || batch.view.size() != 2) {
            return false;
        }
        const auto & slot = slots.front();
        if (slot.id != 0 || slot.state != SLOT_STATE_GENERATING || !slot.task || !slot.ctx_dft ||
                llama_get_model(slot.ctx_dft) != llama_get_model(ctx_tgt) || slot.spec_is_replay ||
                slot.spec_draft.size() != 1 || !slot.spec_draft_q.empty() || !common_speculative_get_synth_probs(spec.get()).empty() ||
                slot.spec_i_batch.size() != 2 || slot.spec_i_batch[0] != 0 || slot.spec_i_batch[1] != 1) {
            return false;
        }
        const auto & sampling = slot.task->params.sampling;
        return occamy_compact::fixed_sampler(sampling) &&
                occamy_compact::exact_eog_mask(sampling, params_base.sampling.logit_bias_eog, mask);
    }

    bool decode(int32_t & n_batch, int32_t off, occamy_compact::Result & compact) {''')
text = change(text, '''            mv1_scope _mv1(1);
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }''', '''            mv1_scope _mv1(1);
            occamy_compact::Mask mask{};
            if (compact_eligible(off, mask)) {
                ret = llama_occamy_compact_decode(ctx_tgt, batch.view.get(), mask,
                        compact_readback_mode() == compact_mode::shadow, compact);
            } else {
                ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            }
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
                if (compact.requested) {
                    compact_record(compact);
                }
            }''')
text = change(text, '    void post_decode(int32_t n_batch_tokens, int32_t off) {', '    void post_decode(int32_t n_batch_tokens, int32_t off, const occamy_compact::Result & compact) {')
text = change(text, '''                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                }
                slot.spec_i_batch.clear();''', '''                } else if (compact.compact_only) {
                    GGML_ASSERT(!mv2_on && n_draft == 1 && !slot.spec_is_replay && occamy_compact::valid(compact.rows));
                    llama_token bonus = LLAMA_TOKEN_NULL;
                    const auto first = occamy_compact::accept_one(compact, slot.spec_draft[0], [&](llama_token id) {
                        common_sampler_accept(slot.smpl.get(), id, true);
                    }, bonus);
                    accepted.reserve(2);
                    accepted.push_back(first);
                    if (bonus != LLAMA_TOKEN_NULL) {
                        accepted.push_back(bonus);
                    }
                } else {
                    if (compact.shadow && compact.launched && occamy_compact::valid(compact.rows)) {
                        // The admitted chain is stateless when sampling without accept.
                        // Check both live rows, including row 1 after a rejection.
                        for (int i = 0; i < 2; ++i) {
                            const auto id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch[i]);
                            ++g_compact.shadow_rows;
                            if (id != compact.rows[i].id) {
                                ++g_compact.shadow_mismatches;
                                throw std::runtime_error("compact readback shadow choice disagreed with the original sampler");
                            }
                        }
                    }
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                }
                slot.spec_i_batch.clear();''')
(here/'server-context.cpp').write_text(text)
(here/'server-context.cpp.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), text.splitlines(True), 'a/tools/server/server-context.cpp','b/tools/server/server-context.cpp')))
