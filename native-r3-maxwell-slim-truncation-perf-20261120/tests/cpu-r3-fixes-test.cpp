#include "compact-readback.h"
#include "server-common.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Exact rate-limit predicate from T1 in ggml-cuda.cu
static bool sf2_should_log(std::atomic<uint64_t> & counter, uint64_t & out_k) {
    const uint64_t k = ++counter;
    out_k = k;
    return k <= 16 || (k & (k - 1)) == 0;
}

struct MockCompletionToken {
    llama_token tok = -1;
    std::string text_to_send;
};

struct MockPartialEvent {
    llama_token tok = -1;
    std::string content;
    size_t      n_decoded = 0;
};

struct MockSlot {
    std::string                      generated_text;
    std::vector<llama_token>         generated_tokens;
    std::vector<MockCompletionToken> generated_token_probs;
    std::vector<MockPartialEvent>    partials;
    size_t                           n_sent_text = 0;
    size_t                           n_gen = 0;
    size_t                           n_predict = 0;
    bool                             has_next_token = true;
    bool                             stream = true;
    std::string                      stop_word;

    void truncate(size_t n) {
        if (n < generated_text.size()) {
            generated_text.resize(n);
        }
    }

    size_t find_stopping_strings(const std::string & text, size_t /*last_len*/, bool full) const {
        if (stop_word.empty()) {
            return std::string::npos;
        }
        if (full) {
            return text.find(stop_word);
        }
        for (size_t k = 1; k < stop_word.size(); ++k) {
            if (text.size() >= k && text.compare(text.size() - k, k, stop_word, 0, k) == 0) {
                return text.size() - k;
            }
        }
        return std::string::npos;
    }

    void add_token(const MockCompletionToken & t) {
        generated_token_probs.push_back(t);
    }

    void send_partial_response(const MockCompletionToken & tkn) {
        const size_t start = n_sent_text - tkn.text_to_send.size();
        const std::string piece = generated_text.substr(start, tkn.text_to_send.size());
        // Delivered SSE content must never end with broken UTF-8 bytes
        assert(validate_utf8(piece) == piece.size());
        partials.push_back({ tkn.tok, piece, n_gen });
    }
};

// Mirrors server-context.cpp process_token() including T3 fix
static bool run_process_token(MockCompletionToken result, MockSlot & slot, bool is_eog = false) {
    const std::string token_str = result.text_to_send;
    slot.generated_text += token_str;
    slot.generated_tokens.push_back(result.tok);
    slot.has_next_token = true;

    const bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

    if (!incomplete) {
        size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());
        const std::string str_test = slot.generated_text.substr(pos);
        bool send_text = true;

        size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
        if (stop_pos != std::string::npos) {
            slot.truncate(pos + stop_pos);
            pos = std::min(slot.n_sent_text, slot.generated_text.size());
        } else if (slot.has_next_token && !is_eog) {
            stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
            send_text = stop_pos == std::string::npos;
        }

        if (send_text) {
            result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
            slot.n_sent_text += result.text_to_send.size();
        } else {
            result.text_to_send = "";
        }

        slot.add_token(result);
        if (slot.stream) {
            slot.send_partial_response(result);
        }
    }

    if (incomplete) {
        slot.has_next_token = true;
    }

    if (slot.n_gen > 0 && slot.has_next_token && slot.n_gen >= slot.n_predict) {
        slot.has_next_token = false;
    }

    if (is_eog) {
        slot.has_next_token = false;
    }

    if (incomplete) {
        result.text_to_send = "";

        if (!slot.has_next_token) {
            const size_t pos       = std::min(slot.n_sent_text, slot.generated_text.size());
            const size_t valid_end = validate_utf8(slot.generated_text);
            bool send_text         = true;
            size_t flush_end       = valid_end;

            if (valid_end > pos) {
                const std::string str_test = slot.generated_text.substr(pos, valid_end - pos);
                const size_t stop_pos = slot.find_stopping_strings(str_test, str_test.size(), true);
                if (stop_pos != std::string::npos) {
                    slot.truncate(pos + stop_pos);
                    flush_end = std::min(valid_end, pos + stop_pos);
                } else if (!is_eog) {
                    const size_t partial_stop = slot.find_stopping_strings(str_test, str_test.size(), false);
                    send_text = partial_stop == std::string::npos;
                }
            }

            if (send_text && flush_end > pos) {
                result.text_to_send = slot.generated_text.substr(pos, flush_end - pos);
                slot.n_sent_text    = flush_end;
            }
        }

        slot.add_token(result);
        if (slot.stream) {
            slot.send_partial_response(result);
        }
    }

    return slot.has_next_token;
}

static void test_t1_sf2_rate_limit() {
    std::atomic<uint64_t> ctr{0};
    std::vector<uint64_t> logged;
    for (int i = 0; i < 1024; ++i) {
        uint64_t k = 0;
        if (sf2_should_log(ctr, k)) {
            logged.push_back(k);
        }
    }
    const std::vector<uint64_t> expected = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        32, 64, 128, 256, 512, 1024
    };
    assert(logged == expected);
    std::printf("PASS test_t1_sf2_rate_limit: 1024 events -> %zu logs\n", logged.size());
}

static void test_t3_utf8_final_token_flush() {
    // Case A: 3071 ASCII tokens + token 3072 ending in incomplete "\xE2\x80" hitting n_predict=3072
    {
        MockSlot slot;
        slot.n_predict = 3072;
        for (int i = 1; i <= 3071; ++i) {
            slot.n_gen = (size_t) i;
            assert(run_process_token({ 100 + i, "a" }, slot));
        }
        assert(slot.partials.size() == 3071);
        slot.n_gen = 3072;
        assert(!run_process_token({ 9999, "\xE2\x80" }, slot));
        assert(slot.partials.size() == 3072);
        assert(slot.generated_token_probs.size() == 3072);
        assert(slot.generated_tokens.size() == 3072);
        assert(slot.partials.back().tok == 9999);
        assert(slot.partials.back().content.empty());
        assert(slot.partials.back().n_decoded == 3072);
        assert(slot.n_sent_text == 3071);
    }

    // Case B: mid-stream incomplete byte "\xE2" emits token ID with empty text, then final token completes "\x80\x94!" + leaves new "\xE2\x80"
    {
        MockSlot slot;
        slot.n_predict = 3;
        slot.n_gen = 1;
        assert(run_process_token({ 1, "hi " }, slot));
        assert(slot.partials.size() == 1);

        slot.n_gen = 2;
        assert(run_process_token({ 2, "\xE2" }, slot));
        assert(slot.partials.size() == 2); // token 2 emitted with empty content while UTF-8 byte held
        assert(slot.partials.back().tok == 2);
        assert(slot.partials.back().content.empty());

        slot.n_gen = 3;
        assert(!run_process_token({ 3, "\x80\x94!\xE2\x80" }, slot));
        assert(slot.partials.size() == 3); // token 3 delivered with valid prefix "\xE2\x80\x94!"
        assert(slot.generated_token_probs.size() == 3);
        assert(slot.generated_tokens.size() == 3);
        assert(slot.partials.back().tok == 3);
        assert(slot.partials.back().content == "\xE2\x80\x94!");
        assert(slot.n_sent_text == 7); // "hi " (3) + "\xE2\x80\x94!" (4)
    }

    // Case C: valid prefix contains stopping string while trailing bytes are incomplete
    {
        MockSlot slot;
        slot.n_predict = 2;
        slot.stop_word = "STOP";
        slot.n_gen = 1;
        assert(run_process_token({ 10, "ok " }, slot));
        slot.n_gen = 2;
        assert(!run_process_token({ 11, "STOPjunk\xE2\x80" }, slot));
        assert(slot.partials.size() == 2);
        assert(slot.partials.back().content.empty());
        assert(slot.generated_text == "ok ");
    }

    std::printf("PASS test_t3_utf8_final_token_flush: 3072/3072 delivered, valid UTF-8 prefix flushed, stop-truncated\n");
}

static bool validate_compact_mask_admission(const occamy_compact::Mask & mask, llama_token n_vocab) {
    bool admitted = true;
    bool seen_sentinel = false;
    llama_token prev = -1;
    for (const llama_token id : mask) {
        if (id == occamy_compact::vocab_size) {
            seen_sentinel = true;
        } else if (seen_sentinel || id < 0 || id >= n_vocab || id <= prev) {
            admitted = false;
            break;
        } else {
            prev = id;
        }
    }
    return admitted;
}

static void test_a2_1row_compact_invariants() {
    occamy_compact::Result r;
    r.rows[0] = { 1234, 0 };
    // rows[1] stays default {LLAMA_TOKEN_NULL, empty} for 1-row draft
    assert(occamy_compact::valid_row(r.rows[0]));
    assert(!occamy_compact::valid(r.rows));

    occamy_compact::Mask dft_mask;
    dft_mask.fill(occamy_compact::vocab_size);
    assert(validate_compact_mask_admission(dft_mask, occamy_compact::vocab_size));

    std::printf("PASS test_a2_1row_compact_invariants: valid_row=1 valid_2row=0 sentinel_mask=admitted\n");
}

} // namespace

int main() {
    test_t1_sf2_rate_limit();
    test_t3_utf8_final_token_flush();
    test_a2_1row_compact_invariants();
    std::printf("ALL R3 FIX UNIT TESTS PASSED\n");
    return 0;
}
