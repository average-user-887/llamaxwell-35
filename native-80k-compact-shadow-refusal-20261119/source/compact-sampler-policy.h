#pragma once

#include "common.h"
#include "compact-readback.h"

#include <cmath>

namespace occamy_compact {
// Restrict optimization to the observed fixed sampler order and effective values.
// Filters following logit bias retain the unique maximum. Disabled filters cannot
// carry row-dependent state into the bonus choice.
inline bool fixed_sampler(const common_params_sampling & p) {
    static const common_params_sampling defaults;
    return p.n_probs == 0 && !p.backend_sampling && p.grammar.empty() && !p.grammar_lazy &&
            !p.reasoning_control && p.reasoning_budget_tokens < 0 && p.ignore_eos &&
            p.temp == 0.0f && p.dynatemp_range == 0.0f && p.mirostat == 0 &&
            p.samplers == defaults.samplers && p.penalty_repeat == defaults.penalty_repeat &&
            p.penalty_freq == 0.0f && p.penalty_present == 0.0f && p.dry_multiplier == 0.0f &&
            p.top_n_sigma == defaults.top_n_sigma && p.top_k == defaults.top_k && p.typ_p == defaults.typ_p &&
            p.top_p == defaults.top_p && p.min_p == defaults.min_p && p.min_keep == defaults.min_keep &&
            p.xtc_probability == 0.0f;
}

inline bool exact_eog_mask(const common_params_sampling & p, const std::vector<llama_logit_bias> & expected, Mask & mask) {
    if (p.logit_bias.size() != mask_size || expected.size() != mask_size) {
        return false;
    }
    for (size_t i = 0; i < mask.size(); ++i) {
        const auto & actual = p.logit_bias[i];
        if (actual.token != expected[i].token || actual.bias != -INFINITY || expected[i].bias != -INFINITY) {
            return false;
        }
        mask[i] = actual.token;
    }
    return true;
}
} // namespace occamy_compact
