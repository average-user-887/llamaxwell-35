#include "common.h"
#include "speculative.h"
#include "compact-readback.h"
#include "llama.h"
#include "llama-ext.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int32_t kNBatch = 2048;
constexpr int32_t kNEmbd  = 16;
constexpr int32_t kNVocab = occamy_compact::vocab_size;

struct MockContext {
    bool is_draft = false;
};

struct MockModel {};
struct MockVocab {};
struct MockMemory {};

static MockModel   g_model;
static MockVocab   g_vocab;
static MockMemory  g_mem_tgt;
static MockMemory  g_mem_dft;
static MockContext g_ctx_tgt{false};
static MockContext g_ctx_dft{true};

struct RecordedRow {
    llama_token        id = LLAMA_TOKEN_NULL;
    llama_pos          pos = -1;
    llama_seq_id       seq_id = 0;
    bool               output = false;
    std::vector<float> state;
};

enum DecodeKind {
    DECODE_PROCESS = 1,
    DECODE_CACHE_CATCH_UP = 2,
    DECODE_COMPACT = 3,
    DECODE_LAST_QUERY = 4,
    DECODE_COMPACT_LAST_QUERY = 5,
};

struct RecordedCall {
    DecodeKind               kind;
    std::vector<RecordedRow> rows;
};

static llama_pos                 g_dft_pos_max = 6175;
static std::vector<float>        g_tgt_nextn;
static std::vector<float>        g_dft_nextn;
static std::vector<float>        g_dft_logits;
static std::vector<RecordedCall> g_calls;
static int                       g_fail_on_call = -1;
static int                       g_set_sampler_calls = 0;

// Compact + last-query mock control for Sub-card A2 tests
static bool                   g_compact_launched = true;
static bool                   g_compact_only = true;
static occamy_compact::Choice g_compact_choice{42, 0};
static struct llama_context * g_last_query_enabled_ctx = nullptr;
static struct llama_context * g_last_query_tgt_seen = nullptr;

static float expected_h_val(llama_pos tgt_pos, int32_t d) {
    return (float) tgt_pos * 100.0f + (float) d * 0.25f + 0.5f;
}

static void set_tgt_nextn_positions(llama_pos pos_start, int32_t n_rows) {
    g_tgt_nextn.resize((size_t) n_rows * kNEmbd);
    for (int32_t i = 0; i < n_rows; ++i) {
        for (int32_t d = 0; d < kNEmbd; ++d) {
            g_tgt_nextn[(size_t) i * kNEmbd + d] = expected_h_val(pos_start + i, d);
        }
    }
}

static void set_dft_logits_argmax(llama_token peak_id, float peak_score) {
    g_dft_logits.assign((size_t) kNVocab, -10.0f);
    g_dft_logits[peak_id] = peak_score;
    g_dft_nextn.assign((size_t) 8 * kNEmbd, 1.0f);
}

} // namespace

struct llama_batch_ext {
    size_t n_tokens_max = kNBatch;
    std::vector<RecordedRow> rows;
};

extern "C" {

uint32_t llama_n_batch(const struct llama_context * /*ctx*/) {
    return (uint32_t) kNBatch;
}

const struct llama_model * llama_get_model(const struct llama_context * /*ctx*/) {
    return reinterpret_cast<const struct llama_model *>(&g_model);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    const auto * mc = reinterpret_cast<const MockContext *>(ctx);
    return reinterpret_cast<llama_memory_t>(mc->is_draft ? &g_mem_dft : &g_mem_tgt);
}

llama_pos llama_memory_seq_pos_max(llama_memory_t /*mem*/, llama_seq_id /*seq_id*/) {
    return g_dft_pos_max;
}

bool llama_memory_seq_rm(llama_memory_t /*mem*/, llama_seq_id /*seq_id*/, llama_pos p0, llama_pos p1) {
    if (p0 <= g_dft_pos_max && (p1 < 0 || p1 > g_dft_pos_max)) {
        g_dft_pos_max = p0 - 1;
    }
    return true;
}

int32_t llama_model_n_embd_out(const struct llama_model * /*model*/) {
    return kNEmbd;
}

int32_t llama_model_n_layer_nextn(const struct llama_model * /*model*/) {
    return 1;
}

enum llama_rope_type llama_model_rope_type(const struct llama_model * /*model*/) {
    return LLAMA_ROPE_TYPE_NONE;
}

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * /*model*/) {
    return reinterpret_cast<const struct llama_vocab *>(&g_vocab);
}

int32_t llama_vocab_n_tokens(const struct llama_vocab * /*vocab*/) {
    return kNVocab;
}

bool llama_vocab_is_eog(const struct llama_vocab * /*vocab*/, llama_token id) {
    return id == 2;
}

llama_token llama_vocab_eos(const struct llama_vocab * /*vocab*/) {
    return 2;
}

llama_token llama_vocab_bos(const struct llama_vocab * /*vocab*/) {
    return 1;
}

llama_token llama_vocab_sep(const struct llama_vocab * /*vocab*/) {
    return -1;
}

const llama_token * llama_vocab_get_suppress_tokens(const struct llama_vocab * /*vocab*/, int32_t * n_tokens) {
    if (n_tokens) {
        *n_tokens = 0;
    }
    return nullptr;
}

bool llama_set_sampler(struct llama_context * /*ctx*/, llama_seq_id /*seq_id*/, struct llama_sampler * /*smpl*/) {
    ++g_set_sampler_calls;
    return true;
}

float * llama_get_logits_ith(struct llama_context * /*ctx*/, int32_t /*i*/) {
    return g_dft_logits.empty() ? nullptr : g_dft_logits.data();
}

float * llama_get_sampled_probs_ith(struct llama_context * /*ctx*/, int32_t /*i*/) {
    return nullptr;
}

float * llama_get_sampled_logits_ith(struct llama_context * /*ctx*/, int32_t /*i*/) {
    return nullptr;
}

llama_token * llama_get_sampled_candidates_ith(struct llama_context * /*ctx*/, int32_t /*i*/) {
    return nullptr;
}

llama_token llama_get_sampled_token_ith(struct llama_context * /*ctx*/, int32_t /*i*/) {
    return LLAMA_TOKEN_NULL;
}

void llama_synchronize(struct llama_context * /*ctx*/) {
}

struct llama_batch_ext * llama_batch_ext_init(struct llama_context * ctx) {
    auto * b = new llama_batch_ext();
    b->n_tokens_max = (size_t) llama_n_batch(ctx);
    return b;
}

void llama_batch_ext_free(struct llama_batch_ext * batch) {
    delete batch;
}

void llama_batch_ext_clear(struct llama_batch_ext * batch) {
    batch->rows.clear();
}

int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, llama_seq_id seq_id, llama_token id) {
    if (batch->rows.size() >= batch->n_tokens_max) {
        return -1;
    }
    RecordedRow r;
    r.id = id;
    r.seq_id = seq_id;
    batch->rows.push_back(std::move(r));
    return (int32_t) (batch->rows.size() - 1);
}

int32_t llama_batch_ext_add_embd(struct llama_batch_ext * batch, llama_seq_id seq_id, struct llama_embd /*embd*/) {
    if (batch->rows.size() >= batch->n_tokens_max) {
        return -1;
    }
    RecordedRow r;
    r.id = LLAMA_TOKEN_NULL;
    r.seq_id = seq_id;
    batch->rows.push_back(std::move(r));
    return (int32_t) (batch->rows.size() - 1);
}

bool llama_batch_ext_add_seq(struct llama_batch_ext * /*batch*/, int32_t /*idx*/, llama_seq_id /*seq_id*/) {
    return true;
}

bool llama_batch_ext_set_pos(struct llama_batch_ext * batch, int32_t idx, const llama_pos * pos) {
    if (idx < 0 || idx >= (int32_t) batch->rows.size()) {
        return false;
    }
    batch->rows[idx].pos = pos[0];
    return true;
}

bool llama_batch_ext_set_embd_token(struct llama_batch_ext * /*batch*/, int32_t /*idx*/, struct llama_embd /*embd*/) {
    return true;
}

bool llama_batch_ext_set_embd_state(struct llama_batch_ext * batch, int32_t idx, struct llama_embd embd) {
    if (idx < 0 || idx >= (int32_t) batch->rows.size()) {
        return false;
    }
    batch->rows[idx].state.assign(embd.data, embd.data + embd.n_rows * embd.n_embd);
    return true;
}

bool llama_batch_ext_set_output_logits(struct llama_batch_ext * batch, int32_t idx, bool value) {
    if (idx < 0 || idx >= (int32_t) batch->rows.size()) {
        return false;
    }
    batch->rows[idx].output = value;
    return true;
}

bool llama_batch_ext_set_output_embd(struct llama_batch_ext * batch, int32_t idx, bool value) {
    return llama_batch_ext_set_output_logits(batch, idx, value);
}

static int32_t record_decode(DecodeKind kind, struct llama_batch_ext * batch) {
    const int call_idx = (int) g_calls.size();
    assert(batch->rows.size() <= (size_t) kNBatch);
    g_calls.push_back({ kind, batch->rows });
    if (g_fail_on_call == call_idx) {
        return -1;
    }
    for (const auto & r : batch->rows) {
        if (r.pos > g_dft_pos_max) {
            g_dft_pos_max = r.pos;
        }
    }
    return 0;
}

int32_t llama_process(struct llama_context * /*ctx*/, enum llama_process_type /*type*/, struct llama_batch_ext * batch) {
    return record_decode(DECODE_PROCESS, batch);
}

} // extern "C"

struct llama_context * llama_get_ctx_other(struct llama_context * /*ctx*/) {
    return nullptr;
}

void llama_set_embeddings_nextn(struct llama_context * /*ctx*/, bool /*value*/, bool /*masked*/) {
}

void llama_set_nextn_layer_offset(struct llama_context * /*ctx*/, int32_t /*offset*/) {
}

float * llama_get_embeddings_nextn(struct llama_context * /*ctx*/) {
    return g_tgt_nextn.empty() ? nullptr : g_tgt_nextn.data();
}

float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i) {
    const auto * mc = reinterpret_cast<const MockContext *>(ctx);
    if (mc->is_draft) {
        return g_dft_nextn.data();
    }
    return g_tgt_nextn.data() + (size_t) i * kNEmbd;
}

bool llama_batch_ext_set_decision_order(struct llama_batch_ext * /*batch*/, int32_t /*idx*/, enum llama_decision_order /*order*/) {
    return true;
}

int32_t llama_mtp_cache_catch_up(struct llama_context * /*ctx*/, struct llama_batch_ext * batch) {
    return record_decode(DECODE_CACHE_CATCH_UP, batch);
}

static void fill_mock_compact_result(const occamy_compact::Mask & mask, bool shadow, occamy_compact::Result & result) {
    for (const llama_token m : mask) {
        assert(m == occamy_compact::vocab_size);
    }
    result = {};
    result.requested = true;
    result.shadow = shadow;
    result.launched = g_compact_launched;
    result.compact = g_compact_launched ? sizeof(occamy_compact::Choice) : 0;
    result.rows[0] = g_compact_choice;
    result.compact_only = g_compact_launched && !shadow && g_compact_only &&
            occamy_compact::valid_row(result.rows[0]);
}

int32_t llama_occamy_compact_decode(llama_context * /*ctx*/, llama_batch_ext * batch,
        const occamy_compact::Mask & mask, bool shadow, occamy_compact::Result & result) {
    fill_mock_compact_result(mask, shadow, result);
    return record_decode(DECODE_COMPACT, batch);
}

void llama_mtp_enable_last_query(struct llama_context * ctx) {
    g_last_query_enabled_ctx = ctx;
}

int32_t llama_mtp_draft_last_query(struct llama_context * ctx, struct llama_context * target, struct llama_batch_ext * batch) {
    assert(target != nullptr && ctx != target);
    g_last_query_tgt_seen = target;
    return record_decode(DECODE_LAST_QUERY, batch);
}

int32_t llama_occamy_compact_draft_last_query(llama_context * ctx, llama_context * target, llama_batch_ext * batch,
        const occamy_compact::Mask & mask, bool shadow, occamy_compact::Result & result) {
    assert(target != nullptr && ctx != target);
    g_last_query_tgt_seen = target;
    fill_mock_compact_result(mask, shadow, result);
    return record_decode(DECODE_COMPACT_LAST_QUERY, batch);
}

static void check_rows(const std::vector<RecordedRow> & all_rows, llama_pos pos0, int32_t total, llama_pos first_h_pos_base) {
    assert((int32_t) all_rows.size() == total);
    for (int32_t i = 0; i < total; ++i) {
        const llama_pos p = pos0 + i;
        assert(all_rows[i].id == 10000 + p);
        assert(all_rows[i].pos == p);
        assert(all_rows[i].seq_id == 0);
        assert((int32_t) all_rows[i].state.size() == kNEmbd);
        for (int32_t d = 0; d < kNEmbd; ++d) {
            const float exp = (p == pos0)
                ? 0.0f
                : expected_h_val(first_h_pos_base + i - 1, d);
            assert(std::fabs(all_rows[i].state[d] - exp) < 1e-6f);
        }
    }
}

static void test_direct_common_batch_cap_and_subbatch() {
    auto * ctx_dft = reinterpret_cast<struct llama_context *>(&g_ctx_dft);
    common_batch batch(ctx_dft);

    const int32_t total = 2 + kNBatch; // 2050
    std::vector<float> states((size_t) total * kNEmbd);
    for (int32_t i = 0; i < total; ++i) {
        const llama_pos p = 6176 + i;
        const int32_t idx = batch.add(10000 + p, p, 0, false);
        for (int32_t d = 0; d < kNEmbd; ++d) {
            states[(size_t) i * kNEmbd + d] = expected_h_val(p, d);
        }
        assert(batch.set_embd_state(idx, { states.data() + (size_t) i * kNEmbd, 1, (size_t) kNEmbd }));
    }
    assert(batch.size() == 2050);

    {
        llama_batch_ext * raw = llama_batch_ext_init(ctx_dft);
        for (int32_t i = 0; i < kNBatch; ++i) {
            assert(llama_batch_ext_add_token(raw, 0, 100 + i) == i);
        }
        assert(llama_batch_ext_add_token(raw, 0, 9999) == -1);
        llama_batch_ext_free(raw);
    }

    {
        std::fflush(nullptr);
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            if (std::freopen("/dev/null", "w", stderr) == nullptr) {
                _exit(2);
            }
            (void) batch.get();
            _exit(0);
        }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    }

    std::vector<RecordedRow> combined;
    for (int32_t off = 0; off < batch.size(); off += kNBatch) {
        const int32_t n_cur = std::min<int32_t>(kNBatch, batch.size() - off);
        llama_batch_ext * sub = batch.get_sub_batch(off, n_cur);
        assert((int32_t) sub->rows.size() == n_cur);
        combined.insert(combined.end(), sub->rows.begin(), sub->rows.end());
    }
    assert((int32_t) combined.size() == total);
    for (int32_t i = 0; i < total; ++i) {
        const llama_pos p = 6176 + i;
        assert(combined[i].id == 10000 + p);
        assert(combined[i].pos == p);
        assert((int32_t) combined[i].state.size() == kNEmbd);
        for (int32_t d = 0; d < kNEmbd; ++d) {
            assert(std::fabs(combined[i].state[d] - expected_h_val(p, d)) < 1e-6f);
        }
    }
    std::printf("PASS test_direct_common_batch_cap_and_subbatch: total=%d abort_on_2050=SIGABRT subbatches=[2048,2]\n",
                total);
}

static void run_e2e_speculative_scenario(bool cache_only_mode) {
    setenv("LLAMA_MTP_CACHE_ONLY", cache_only_mode ? "1" : "0", 1);
    setenv("LLAMA_MTP_MERGE_REEVAL", "1", 1);
    setenv("LLAMA_OCCAMY_COMPACT_DRAFT", "0", 1);

    g_dft_pos_max = 6175;
    g_calls.clear();
    g_fail_on_call = -1;

    auto * ctx_tgt = reinterpret_cast<struct llama_context *>(&g_ctx_tgt);
    auto * ctx_dft = reinterpret_cast<struct llama_context *>(&g_ctx_dft);

    common_params_speculative sp;
    sp.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    sp.draft.ctx_tgt = ctx_tgt;
    sp.draft.ctx_dft = ctx_dft;
    sp.draft.n_max = 1;
    sp.draft.n_min = 0;
    sp.draft.p_min = 0.0f;
    sp.draft.backend_sampling = false;

    common_speculative * spec = common_speculative_init(sp, 1);
    assert(spec != nullptr);

    common_speculative_begin(spec, 0, llama_tokens{});

    common_batch verify_b(ctx_tgt);
    verify_b.add(16176, 6176, 0, true);
    verify_b.add(16177, 6177, 0, true);
    set_tgt_nextn_positions(6176, 2);
    assert(common_speculative_process(spec, verify_b));
    assert(g_calls.empty());
    assert(g_dft_pos_max == 6175);

    common_batch prefill_b(ctx_tgt);
    for (int32_t i = 0; i < kNBatch; ++i) {
        const llama_pos p = 6178 + i;
        prefill_b.add(10000 + p, p, 0, true);
    }
    set_tgt_nextn_positions(6178, kNBatch);
    assert(common_speculative_process(spec, prefill_b));

    assert(g_calls.size() == 2);
    const DecodeKind exp_kind = cache_only_mode ? DECODE_CACHE_CATCH_UP : DECODE_PROCESS;
    assert(g_calls[0].kind == exp_kind && g_calls[0].rows.size() == 2048);
    assert(g_calls[1].kind == exp_kind && g_calls[1].rows.size() == 2);
    assert(g_dft_pos_max == 8225);

    std::vector<RecordedRow> all_rows = g_calls[0].rows;
    all_rows.insert(all_rows.end(), g_calls[1].rows.begin(), g_calls[1].rows.end());
    check_rows(all_rows, 6176, 2050, 6176);

    g_dft_pos_max = 6175;
    g_calls.clear();
    g_fail_on_call = -1;
    set_tgt_nextn_positions(6176, 2);
    assert(common_speculative_process(spec, verify_b));
    assert(g_calls.empty());
    g_fail_on_call = 1;
    set_tgt_nextn_positions(6178, kNBatch);
    assert(!common_speculative_process(spec, prefill_b));
    assert(g_calls.size() == 2);

    common_speculative_free(spec);
    std::printf("PASS e2e_speculative_process (cache_only=%d): calls=[2048,2] pos=[%d..%d] fail_rc_propagated=1\n",
                (int) cache_only_mode, (int) all_rows.front().pos, (int) all_rows.back().pos);
}

static void test_subcard_a2_compact_draft() {
    auto * ctx_tgt = reinterpret_cast<struct llama_context *>(&g_ctx_tgt);
    auto * ctx_dft = reinterpret_cast<struct llama_context *>(&g_ctx_dft);

    common_params_speculative sp;
    sp.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    sp.draft.ctx_tgt = ctx_tgt;
    sp.draft.ctx_dft = ctx_dft;
    sp.draft.n_max = 1;
    sp.draft.n_min = 0;
    sp.draft.p_min = 0.0f;
    sp.draft.backend_sampling = true; // verify compact draft skips attaching backend_chains

    // 1. Mode = "1" (compact-only hit, no fallback)
    {
        setenv("LLAMA_MTP_MERGE_REEVAL", "0", 1);
        setenv("LLAMA_OCCAMY_COMPACT_DRAFT", "1", 1);
        g_set_sampler_calls = 0;
        g_calls.clear();
        g_fail_on_call = -1;
        g_dft_pos_max = 99;
        g_compact_launched = true;
        g_compact_only = true;
        g_compact_choice = { 1234, 0 };
        set_dft_logits_argmax(9999, 10.0f); // different from compact choice -> proves compact-only skipped sampler!

        common_speculative * spec = common_speculative_init(sp, 1);
        assert(spec != nullptr);
        assert(g_set_sampler_calls == 0);

        llama_tokens dummy_prompt;
        llama_tokens drafted;
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max = 1;
        dp.pos0 = 100;
        dp.id_last = 77;
        dp.prompt = &dummy_prompt;
        dp.result = &drafted;
        common_speculative_draft(spec);

        assert(g_calls.size() == 1 && g_calls[0].kind == DECODE_COMPACT);
        assert(g_calls[0].rows.size() == 1 && g_calls[0].rows[0].output);
        assert(drafted.size() == 1 && drafted[0] == 1234);

        // Fallback on tie (flags != 0 -> compact_only=false -> runs CPU sampler on logits peak 9999)
        g_calls.clear();
        g_compact_choice = { 1234, occamy_compact::tie };
        drafted.clear();
        dp.drafting = true;
        dp.pos0 = 101;
        dp.id_last = 1234;
        dp.result = &drafted;
        common_speculative_draft(spec);

        assert(g_calls.size() == 1 && g_calls[0].kind == DECODE_COMPACT);
        assert(drafted.size() == 1 && drafted[0] == 9999);

        common_speculative_free(spec);
    }

    // 2. Mode = "shadow" (always runs stock sampler, compares compact_choice vs stock peak)
    {
        setenv("LLAMA_OCCAMY_COMPACT_DRAFT", "shadow", 1);
        g_set_sampler_calls = 0;
        g_calls.clear();
        g_compact_launched = true;
        g_compact_only = false;
        g_compact_choice = { 5555, 0 };
        set_dft_logits_argmax(5555, 12.0f);

        common_speculative * spec = common_speculative_init(sp, 1);
        assert(spec != nullptr);
        assert(g_set_sampler_calls == 0);

        llama_tokens dummy_prompt;
        llama_tokens drafted;
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max = 1;
        dp.pos0 = 200;
        dp.id_last = 88;
        dp.prompt = &dummy_prompt;
        dp.result = &drafted;
        common_speculative_draft(spec);

        assert(g_calls.size() == 1 && g_calls[0].kind == DECODE_COMPACT);
        assert(drafted.size() == 1 && drafted[0] == 5555);
        common_speculative_free(spec);
    }

    // 3. Mode = "0" (default off -> attaches backend_chains and calls llama_process)
    {
        setenv("LLAMA_OCCAMY_COMPACT_DRAFT", "0", 1);
        g_set_sampler_calls = 0;
        g_calls.clear();
        set_dft_logits_argmax(7777, 15.0f);

        common_speculative * spec = common_speculative_init(sp, 1);
        assert(spec != nullptr);
        assert(g_set_sampler_calls == 1);

        llama_tokens dummy_prompt;
        llama_tokens drafted;
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max = 1;
        dp.pos0 = 300;
        dp.id_last = 99;
        dp.prompt = &dummy_prompt;
        dp.result = &drafted;
        common_speculative_draft(spec);

        assert(g_calls.size() == 1 && g_calls[0].kind == DECODE_PROCESS);
        assert(drafted.size() == 1 && drafted[0] == 7777);
        common_speculative_free(spec);
    }

    std::printf("PASS test_subcard_a2_compact_draft: on=1234 fallback_tie=9999 shadow=5555 off=7777\n");
}

static void test_last_query_compact_draft() {
    auto * ctx_tgt = reinterpret_cast<struct llama_context *>(&g_ctx_tgt);
    auto * ctx_dft = reinterpret_cast<struct llama_context *>(&g_ctx_dft);

    common_params_speculative sp;
    sp.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    sp.draft.ctx_tgt = ctx_tgt;
    sp.draft.ctx_dft = ctx_dft;
    sp.draft.n_max = 1;
    sp.draft.n_min = 0;
    sp.draft.p_min = 0.0f;

    const auto run_case = [&](const char * compact_env, bool backend_sampling,
            DecodeKind exp_kind, llama_token exp_tok, llama_token fallback_tok) {
        setenv("LLAMA_MTP_MERGE_REEVAL", "1", 1);
        setenv("LLAMA_MTP_LAST_QUERY", "1", 1);
        setenv("LLAMA_OCCAMY_COMPACT_DRAFT", compact_env, 1);
        sp.draft.backend_sampling = backend_sampling;
        g_calls.clear();
        g_fail_on_call = -1;
        g_dft_pos_max = 99;
        g_last_query_tgt_seen = nullptr;
        g_compact_launched = true;
        g_compact_only = true;
        g_compact_choice = { exp_tok, 0 };
        set_dft_logits_argmax(fallback_tok, 12.0f);

        common_speculative * spec = common_speculative_init(sp, 1);
        assert(spec != nullptr);
        common_speculative_begin(spec, 0, llama_tokens{});

        common_batch verify_b(ctx_tgt);
        verify_b.add(10100, 100, 0, true);
        verify_b.add(10101, 101, 0, true);
        set_tgt_nextn_positions(100, 2);
        assert(common_speculative_process(spec, verify_b));
        assert(g_calls.empty());

        llama_tokens dummy_prompt;
        llama_tokens drafted;
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max = 1;
        dp.pos0 = 102;
        dp.id_last = 88;
        dp.temp = 0.0f;
        dp.result_q = nullptr;
        dp.prompt = &dummy_prompt;
        dp.result = &drafted;
        common_speculative_draft(spec);

        assert(g_calls.size() == 1 && g_calls[0].kind == exp_kind);
        assert(g_calls[0].rows.size() == 3);
        assert(g_last_query_tgt_seen == ctx_tgt);
        assert(drafted.size() == 1 && drafted[0] == exp_tok);
        common_speculative_free(spec);
    };

    run_case("1", true, DECODE_COMPACT_LAST_QUERY, 2345, 9999);
    run_case("0", false, DECODE_LAST_QUERY, 3456, 3456);
    setenv("LLAMA_MTP_LAST_QUERY", "0", 1);

    std::printf("PASS test_last_query_compact_draft: compact=2345 standard=3456\n");
}

int main() {
    test_direct_common_batch_cap_and_subbatch();

    std::fflush(nullptr);
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        run_e2e_speculative_scenario(true);
        std::fflush(nullptr);
        _exit(0);
    }
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    run_e2e_speculative_scenario(false);
    test_subcard_a2_compact_draft();
    test_last_query_compact_draft();

    std::printf("ALL CARD E + SUB-CARD A2 TESTS PASSED\n");
    return 0;
}
