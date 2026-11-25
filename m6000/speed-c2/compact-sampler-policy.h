#pragma once

#include "common.h"
#include "compact-readback.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace occamy_compact {

inline bool native_mtp_types(const std::vector<common_speculative_type> & types) {
    if (types.size() != 1 && types.size() != 2) {
        return false;
    }
    return std::count(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == 1 &&
            std::count(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE) == static_cast<ptrdiff_t>(types.size() - 1);
}

inline bool greedy_safe_sampler_type(common_sampler_type type) {
    switch (type) {
        case COMMON_SAMPLER_TYPE_DRY:
        case COMMON_SAMPLER_TYPE_TOP_K:
        case COMMON_SAMPLER_TYPE_TOP_P:
        case COMMON_SAMPLER_TYPE_MIN_P:
        case COMMON_SAMPLER_TYPE_TYPICAL_P:
        case COMMON_SAMPLER_TYPE_TEMPERATURE:
        case COMMON_SAMPLER_TYPE_XTC:
        case COMMON_SAMPLER_TYPE_PENALTIES:
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
            return true;
        default:
            return false;
    }
}

// Admit greedy chains whose active filters preserve a unique argmax and carry no
// row-dependent state into the bonus choice.
inline bool fixed_sampler(const common_params_sampling & p) {
    const bool valid_samplers = !p.samplers.empty() &&
            (p.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE ||
             (p.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && p.top_k == 1)) &&
            std::all_of(p.samplers.begin(), p.samplers.end(), greedy_safe_sampler_type);

    return p.n_probs == 0 && !p.backend_sampling && p.grammar.empty() && !p.grammar_lazy &&
            !p.reasoning_control && p.reasoning_budget_tokens < 0 &&
            p.temp == 0.0f && p.dynatemp_range == 0.0f && p.mirostat == 0 &&
            valid_samplers &&
            p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f &&
            p.dry_multiplier == 0.0f && p.xtc_probability == 0.0f && p.top_n_sigma <= 0.0f &&
            p.top_k >= 0 && p.min_keep >= 0 &&
            std::isfinite(p.top_p) && p.top_p > 0.0f &&
            std::isfinite(p.min_p) && p.min_p >= 0.0f && p.min_p <= 1.0f &&
            std::isfinite(p.typ_p) && p.typ_p >= 1.0f;
}

inline bool build_compact_mask(const common_params_sampling & p,
        const std::vector<llama_logit_bias> & expected_eog, Mask & mask) {
    if (p.ignore_eos) {
        if (p.logit_bias.size() != mask_size || expected_eog.size() != mask_size) {
            return false;
        }
        llama_token prev = -1;
        for (size_t i = 0; i < mask.size(); ++i) {
            const auto & actual = p.logit_bias[i];
            if (actual.token != expected_eog[i].token ||
                    actual.bias != -INFINITY || expected_eog[i].bias != -INFINITY ||
                    actual.token < 0 || actual.token >= vocab_size || actual.token <= prev) {
                return false;
            }
            prev = actual.token;
            mask[i] = actual.token;
        }
        return true;
    }

    if (p.logit_bias.empty()) {
        mask.fill(vocab_size);
        return true;
    }

    if (p.logit_bias.size() > mask_size) {
        return false;
    }

    std::array<llama_token, mask_size> tokens{};
    size_t n = 0;
    for (const auto & entry : p.logit_bias) {
        if (entry.bias != -INFINITY || entry.token < 0 || entry.token >= vocab_size) {
            return false;
        }
        tokens[n++] = entry.token;
    }

    std::sort(tokens.begin(), tokens.begin() + n);
    n = static_cast<size_t>(std::unique(tokens.begin(), tokens.begin() + n) - tokens.begin());
    mask.fill(vocab_size);
    std::copy(tokens.begin(), tokens.begin() + n, mask.begin());
    return true;
}

} // namespace occamy_compact
