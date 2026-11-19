#!/usr/bin/env python3
from pathlib import Path
import difflib

here=Path(__file__).resolve().parent
before=text=(here.parent/'break100-compact-model/server-context.cpp').read_text()


def change(old,new):
    global text
    assert text.count(old)==1, old
    text=text.replace(old,new)


change('''        const char * scores = std::getenv("LLAMA_MV2_DUMP");
        if (!value || (scores && scores[0])) {''','''        if (!value) {''')
change('''// The only writer is the server worker. Report cumulative byte counts per response.''','''static bool compact_score_observer() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MV2_DUMP");
        return value && value[0];
    }();
    return enabled;
}

enum class compact_guard : size_t { ready, observer, types, rollback, shape, model, draft, sampler, mask, count };
static std::array<unsigned long long, size_t(compact_guard::count)> g_compact_guards{};

// The only writer is the server worker. Report cumulative byte counts per response.''')
change('''        if (compact_readback_mode() != compact_mode::off) {
            fprintf(stderr, "compact_readback:''','''        if (compact_readback_mode() != compact_mode::off) {
            fprintf(stderr, "compact_guard: ready=%llu observer=%llu types=%llu rollback=%llu shape=%llu model=%llu draft=%llu sampler=%llu mask=%llu\\n",
                    g_compact_guards[0], g_compact_guards[1], g_compact_guards[2], g_compact_guards[3], g_compact_guards[4],
                    g_compact_guards[5], g_compact_guards[6], g_compact_guards[7], g_compact_guards[8]);
            fprintf(stderr, "compact_readback:''')
start=text.index('    bool compact_eligible(')
end=text.index('    bool decode(int32_t & n_batch',start)
text=text[:start]+'''    bool compact_eligible(int32_t off, occamy_compact::Mask & mask) const {
        if (compact_readback_mode() == compact_mode::off || !spec || slots.size() != 1) {
            return false;
        }
        const auto & slot = slots.front();
        if (slot.state != SLOT_STATE_GENERATING || !slot.task || slot.spec_draft.empty()) {
            return false;
        }
        const auto reason = [&] {
            if (compact_score_observer()) {
                return compact_guard::observer;
            }
            if (!occamy_compact::native_mtp_types(params_base.speculative.types)) {
                return compact_guard::types;
            }
            if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS || llama_n_rs_seq(ctx_tgt) < 1) {
                return compact_guard::rollback;
            }
            if (slot.id != 0 || off != 0 || batch.size() != 2 || batch.view.size() != 2 ||
                    slot.spec_i_batch.size() != 2 || slot.spec_i_batch[0] != 0 || slot.spec_i_batch[1] != 1) {
                return compact_guard::shape;
            }
            if (!slot.ctx_dft || llama_get_model(slot.ctx_dft) != llama_get_model(ctx_tgt)) {
                return compact_guard::model;
            }
            if (slot.spec_is_replay || slot.spec_draft.size() != 1 || !slot.spec_draft_q.empty() ||
                    !common_speculative_get_synth_probs(spec.get()).empty()) {
                return compact_guard::draft;
            }
            const auto & sampling = slot.task->params.sampling;
            if (!occamy_compact::fixed_sampler(sampling)) {
                return compact_guard::sampler;
            }
            if (!occamy_compact::exact_eog_mask(sampling, params_base.sampling.logit_bias_eog, mask)) {
                return compact_guard::mask;
            }
            return compact_guard::ready;
        }();
        ++g_compact_guards[size_t(reason)];
        return reason == compact_guard::ready;
    }

'''+text[end:]
(here/'server-context.cpp').write_text(text)
(here/'server-context.r1.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True),text.splitlines(True),'a/tools/server/server-context.cpp','b/tools/server/server-context.cpp')))
