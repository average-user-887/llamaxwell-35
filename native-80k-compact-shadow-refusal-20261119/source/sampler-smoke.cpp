#include "compact-sampler-policy.h"

#include <cassert>
#include <cstdio>

int main() {
    using namespace occamy_compact;
    common_params_sampling p;
    p.temp = 0;
    p.ignore_eos = true;
    assert(fixed_sampler(p));
    for (const auto mutation : std::vector<void (*)(common_params_sampling &)>{
            [](auto & a) { a.n_probs = 1; }, [](auto & a) { a.backend_sampling = true; },
            [](auto & a) { a.ignore_eos = false; }, [](auto & a) { a.temp = 0.1f; },
            [](auto & a) { a.dynatemp_range = 0.1f; }, [](auto & a) { a.mirostat = 1; },
            [](auto & a) { a.samplers.pop_back(); }, [](auto & a) { a.penalty_repeat = 1.01f; },
            [](auto & a) { a.penalty_freq = 0.1f; }, [](auto & a) { a.penalty_present = 0.1f; },
            [](auto & a) { a.dry_multiplier = 0.1f; }, [](auto & a) { a.top_k = 1; },
            [](auto & a) { a.typ_p = 0.9f; }, [](auto & a) { a.xtc_probability = 0.1f; },
            [](auto & a) { a.grammar_lazy = true; }, [](auto & a) { a.reasoning_budget_tokens = 0; },
            [](auto & a) { a.reasoning_control = true; },
        }) {
        auto q=p;
        mutation(q);
        assert(!fixed_sampler(q));
    }
    Mask mask{};
    assert(!exact_eog_mask(p, {}, mask));
    const std::vector<llama_logit_bias> eog{{914, -INFINITY},{921, -INFINITY},{945, -INFINITY},{962, -INFINITY},{999, -INFINITY}};
    p.logit_bias = eog;
    assert(exact_eog_mask(p,eog,mask));
    p.logit_bias[3].bias = -100;
    assert(!exact_eog_mask(p,eog,mask));
    p.logit_bias = eog;
    p.logit_bias.push_back({1000,-INFINITY});
    assert(!exact_eog_mask(p,eog,mask));
    Result result;
    assert(!valid(result.rows));
    result.rows = {{{17,0},{39,0}}};
    assert(valid(result.rows));
    for (unsigned flag : {tie,nonfinite,empty}) {
        result.rows[1].flags=flag;
        assert(!valid(result.rows));
    }
    result.rows[1].flags=0;
    std::vector<llama_token> seen;
    const auto accept = [&](llama_token t){seen.push_back(t);};
    llama_token bonus = LLAMA_TOKEN_NULL;
    assert(accept_one(result, 17, accept, bonus) == 17 && bonus == 39 && (seen == llama_tokens{17,39}));
    seen.clear();
    assert(accept_one(result, 18, accept, bonus) == 17 && bonus == LLAMA_TOKEN_NULL && (seen == llama_tokens{17}));
    std::puts("compact fixed sampler, strict mask, invalid result and ordered accept/reject passed");
}
