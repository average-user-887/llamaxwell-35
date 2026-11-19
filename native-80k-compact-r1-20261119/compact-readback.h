#pragma once

#include "llama.h"

#include <array>
#include <cstdint>

namespace occamy_compact {
constexpr int vocab_size = 248320;
constexpr int mask_size = 5;
constexpr size_t raw_bytes = 2 * vocab_size * sizeof(float);
using Mask = std::array<llama_token, mask_size>;

constexpr unsigned tie = 1, nonfinite = 2, empty = 4;
struct Choice { int id; unsigned flags; };
static_assert(sizeof(Choice) == 8);

struct Result {
    std::array<Choice, 2> rows{{{LLAMA_TOKEN_NULL, empty}, {LLAMA_TOKEN_NULL, empty}}};
    bool requested = false;
    bool launched = false;
    bool shadow = false;
    bool compact_only = false;
    size_t raw = 0;
    size_t compact = 0;
};

inline bool valid(const std::array<Choice, 2> & choices) {
    for (const Choice & choice : choices) {
        if (choice.id < 0 || choice.id >= vocab_size || choice.flags) {
            return false;
        }
    }
    return true;
}

// Single-proposal specialization of common_sampler_sample_and_accept_n.
// State advances only for the target choices consumed by the original loop.
template<typename Accept>
llama_token accept_one(const Result & result, llama_token draft, Accept accept, llama_token & bonus) {
    const llama_token first = result.rows[0].id;
    accept(first);
    bonus = LLAMA_TOKEN_NULL;
    if (first == draft) {
        bonus = result.rows[1].id;
        accept(bonus);
    }
    return first;
}
} // namespace occamy_compact

LLAMA_API int32_t llama_occamy_compact_decode(llama_context * ctx, llama_batch_ext * batch,
        const occamy_compact::Mask & mask, bool shadow, occamy_compact::Result & result);
