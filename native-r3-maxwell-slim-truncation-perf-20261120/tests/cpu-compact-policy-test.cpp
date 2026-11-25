#include "compact-sampler-policy.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace occamy_compact;

    // 1. fixed_sampler: default greedy (ignore_eos=false, temp=0.0f) admitted
    common_params_sampling p;
    p.temp = 0.0f;
    assert(!p.ignore_eos);
    assert(fixed_sampler(p));

    // Greedy-safe filters admitted
    {
        auto q = p;
        q.ignore_eos = true;
        assert(fixed_sampler(q));
    }
    {
        auto q = p;
        q.top_k = 0;
        q.top_p = 1.0f;
        q.min_p = 0.0f;
        q.min_keep = 0;
        assert(fixed_sampler(q));
    }
    {
        auto q = p;
        q.top_k = 100;
        q.top_p = 0.8f;
        q.min_p = 0.1f;
        q.min_keep = 1;
        assert(fixed_sampler(q));
    }
    {
        auto q = p;
        q.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
        assert(fixed_sampler(q));
    }
    {
        auto q = p;
        q.samplers = {COMMON_SAMPLER_TYPE_TOP_K};
        q.top_k = 1;
        assert(fixed_sampler(q));
    }

    // Unsafe / non-greedy mutations rejected
    for (const auto mutation : std::vector<void (*)(common_params_sampling &)>{
            [](auto & a) { a.n_probs = 1; },
            [](auto & a) { a.backend_sampling = true; },
            [](auto & a) { a.temp = 0.1f; },
            [](auto & a) { a.temp = -0.1f; },
            [](auto & a) { a.dynatemp_range = 0.1f; },
            [](auto & a) { a.mirostat = 1; },
            [](auto & a) { a.samplers.pop_back(); },
            [](auto & a) { a.samplers.push_back(COMMON_SAMPLER_TYPE_INFILL); },
            [](auto & a) { a.samplers.push_back(COMMON_SAMPLER_TYPE_ADAPTIVE_P); },
            [](auto & a) { a.samplers.clear(); },
            [](auto & a) { a.penalty_repeat = 1.01f; },
            [](auto & a) { a.penalty_freq = 0.1f; },
            [](auto & a) { a.penalty_present = 0.1f; },
            [](auto & a) { a.dry_multiplier = 0.1f; },
            [](auto & a) { a.xtc_probability = 0.1f; },
            [](auto & a) { a.top_n_sigma = 1.0f; },
            [](auto & a) { a.typ_p = 0.9f; },
            [](auto & a) { a.top_k = -1; },
            [](auto & a) { a.top_p = 0.0f; },
            [](auto & a) { a.min_p = 1.1f; },
            [](auto & a) { a.min_p = -0.1f; },
            [](auto & a) { a.min_keep = -1; },
            [](auto & a) { a.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\""); },
            [](auto & a) { a.grammar_lazy = true; },
            [](auto & a) { a.reasoning_budget_tokens = 0; },
            [](auto & a) { a.reasoning_control = true; },
        }) {
        auto q = p;
        mutation(q);
        assert(!fixed_sampler(q));
    }

    // 2. build_compact_mask
    const std::vector<llama_logit_bias> eog{
        {914, -INFINITY}, {921, -INFINITY}, {945, -INFINITY}, {962, -INFINITY}, {999, -INFINITY}
    };
    Mask mask{};

    // (b) !ignore_eos && logit_bias.empty() -> {248320, ...}
    assert(build_compact_mask(p, eog, mask));
    assert((mask == Mask{vocab_size, vocab_size, vocab_size, vocab_size, vocab_size}));

    // (a) ignore_eos=true exact 5-EOG match
    {
        auto q = p;
        q.ignore_eos = true;
        assert(!build_compact_mask(q, eog, mask));
        q.logit_bias = eog;
        assert(build_compact_mask(q, eog, mask));
        assert((mask == Mask{914, 921, 945, 962, 999}));
        q.logit_bias[3].bias = -100.0f;
        assert(!build_compact_mask(q, eog, mask));
        q.logit_bias = eog;
        q.logit_bias.push_back({1000, -INFINITY});
        assert(!build_compact_mask(q, eog, mask));
    }

    // (c) !ignore_eos && 1..5 -INFINITY logit_bias sorted, deduplicated + sentinel padded
    {
        auto q = p;
        q.logit_bias = {{500, -INFINITY}, {120, -INFINITY}, {500, -INFINITY}, {300, -INFINITY}};
        assert(build_compact_mask(q, eog, mask));
        assert((mask == Mask{120, 300, 500, vocab_size, vocab_size}));

        // Exactly 5 unique entries
        q.logit_bias = {{50, -INFINITY}, {10, -INFINITY}, {40, -INFINITY}, {20, -INFINITY}, {30, -INFINITY}};
        assert(build_compact_mask(q, eog, mask));
        assert((mask == Mask{10, 20, 30, 40, 50}));

        // Finite bias rejected
        q.logit_bias = {{50, -10.0f}};
        assert(!build_compact_mask(q, eog, mask));

        // >5 biases rejected
        q.logit_bias = {{1, -INFINITY}, {2, -INFINITY}, {3, -INFINITY}, {4, -INFINITY}, {5, -INFINITY}, {6, -INFINITY}};
        assert(!build_compact_mask(q, eog, mask));

        // Out-of-range token rejected
        q.logit_bias = {{-1, -INFINITY}};
        assert(!build_compact_mask(q, eog, mask));
        q.logit_bias = {{vocab_size, -INFINITY}};
        assert(!build_compact_mask(q, eog, mask));
    }

    // 3. valid_row & valid (both rows must be valid for common_sampler_sample_and_accept_n parity)
    Result result;
    assert(!valid(result.rows));
    result.rows = {{{17, 0}, {39, 0}}};
    assert(valid(result.rows));
    assert(valid_row(result.rows[0]) && valid_row(result.rows[1]));
    for (unsigned flag : {tie, nonfinite, empty}) {
        result.rows[1].flags = flag;
        assert(valid_row(result.rows[0]));
        assert(!valid_row(result.rows[1]));
        assert(!valid(result.rows));
    }
    // Even if row 0 is EOG (914), invalid row 1 must fail valid() and trigger stock fallback
    result.rows = {{{914, 0}, {39, tie}}};
    assert(!valid(result.rows));
    result.rows[1].flags = nonfinite;
    assert(!valid(result.rows));
    result.rows[0] = {914, tie};
    result.rows[1] = {39, 0};
    assert(!valid(result.rows));

    // 4. accept_one parity with common_sampler_sample_and_accept_n for n_draft == 1
    // (draft = {EOG} with first == EOG accepts {EOG, bonus} because i + 1 < draft.size() is false)
    std::vector<llama_token> seen;
    const auto accept = [&](llama_token t) { seen.push_back(t); };
    llama_token bonus = LLAMA_TOKEN_NULL;

    result.rows = {{{914, 0}, {39, 0}}};
    assert(valid(result.rows));
    assert(accept_one(result, 914, accept, bonus) == 914);
    assert(bonus == 39);
    assert((seen == llama_tokens{914, 39}));

    // first == draft != EOG: commits both first and bonus
    seen.clear();
    result.rows = {{{17, 0}, {39, 0}}};
    assert(accept_one(result, 17, accept, bonus) == 17);
    assert(bonus == 39);
    assert((seen == llama_tokens{17, 39}));

    // first != draft: commits first only
    seen.clear();
    assert(accept_one(result, 18, accept, bonus) == 17);
    assert(bonus == LLAMA_TOKEN_NULL);
    assert((seen == llama_tokens{17}));

    // 5. Live llama_sampler_chain verification against policy assumptions
    {
        // Prove typ_p < 1.0f can discard argmax even with min_keep = 1
        std::vector<llama_token_data> cur;
        cur.push_back({0, std::log(0.50f), 0.0f});
        cur.push_back({1, std::log(0.05f), 0.0f});
        for (int i = 2; i < 92; ++i) {
            cur.push_back({i, std::log(0.005f), 0.0f});
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler * typ = llama_sampler_init_typical(0.05f, 1);
        llama_sampler_apply(typ, &cur_p);
        llama_sampler * greedy = llama_sampler_init_greedy();
        llama_sampler_apply(greedy, &cur_p);
        const llama_token chosen = cur_p.data[cur_p.selected].id;
        llama_sampler_free(greedy);
        llama_sampler_free(typ);
        assert(chosen == 1); // typical_p=0.05f dropped token 0 (the argmax) in favor of token 1!
    }

    {
        // Verify admitted chain (logit_bias(-INF) + top_k + top_p + min_p + temp_ext(0) + greedy)
        // matches masked argmax even when min_keep = 0.
        std::vector<llama_logit_bias> biases = {{2, -INFINITY}, {0, -INFINITY}};
        auto q = p;
        q.top_k = 2;
        q.top_p = 0.5f;
        q.min_p = 0.2f;
        q.min_keep = 0;
        q.logit_bias = biases;
        assert(fixed_sampler(q));
        assert(build_compact_mask(q, eog, mask));
        assert((mask == Mask{0, 2, vocab_size, vocab_size, vocab_size}));

        std::vector<llama_token_data> cur = {
            {0, 10.0f, 0.0f},
            {1,  4.0f, 0.0f},
            {2,  9.0f, 0.0f},
            {3,  7.5f, 0.0f},
            {4,  1.0f, 0.0f},
        };
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(5, biases.size(), biases.data()));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(q.top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(q.top_p, q.min_keep));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(q.min_p, q.min_keep));
        llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        llama_sampler_apply(chain, &cur_p);
        assert(cur_p.data[cur_p.selected].id == 3);
        llama_sampler_free(chain);
    }

    std::puts("cpu-compact-policy-test: all assertions passed");
    return 0;
}
