#include "arg.h"
#include "compact-sampler-policy.h"

#include <cassert>
#include <cstdio>

int main(int argc, char ** argv) {
    common_params parsed;
    assert(common_params_parse(argc, argv, parsed, LLAMA_EXAMPLE_SERVER));
    const auto & types=parsed.speculative.types;
    assert((types==std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP}));
    assert(occamy_compact::native_mtp_types(types));
    assert(occamy_compact::native_mtp_types({COMMON_SPECULATIVE_TYPE_DRAFT_MTP}));
    assert(!occamy_compact::native_mtp_types({COMMON_SPECULATIVE_TYPE_NONE}));
    assert(!occamy_compact::native_mtp_types({COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_NGRAM_MOD}));
    assert(!occamy_compact::native_mtp_types({COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_MTP}));
    assert(!occamy_compact::native_mtp_types({COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP}));
    std::puts("compiled runtime parser yields NONE+MTP; exact single-MTP allowlist admits it and rejects four invalid types vectors");
}
