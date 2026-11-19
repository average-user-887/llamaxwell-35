// Exact observer and row-commit excerpts from server-context.cpp.
// Original source lines 120-154.
// MV2 top-8 dump (LLAMA_MV2_DUMP=<file>, LLAMA_MV2_DUMP_LO/HI = n_decoded range, default 0..64; off when unset)
struct mv2_dump_t {
    FILE * f = nullptr;
    long lo = 0, hi = 64;
    mv2_dump_t() {
        const char * p = getenv("LLAMA_MV2_DUMP");
        if (!p || !p[0]) return;
        f = fopen(p, "a");
        if (const char * e = getenv("LLAMA_MV2_DUMP_LO")) lo = atol(e);
        if (const char * e = getenv("LLAMA_MV2_DUMP_HI")) hi = atol(e);
    }
    ~mv2_dump_t() { if (f) fclose(f); }
    bool want(long n) const { return f && n >= lo && n <= hi; }
    // append " idx:id=lp,..." (top-8 log-softmax) for one logits row
    void row(std::string & out, llama_context * ctx, int32_t idx) const {
        const float * lg = llama_get_logits_ith(ctx, idx);
        if (!lg) { out += " [null]"; return; }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        double mx = -1e30;
        for (int i = 0; i < n_vocab; ++i) mx = std::max<double>(mx, lg[i]);
        double z = 0.0;
        for (int i = 0; i < n_vocab; ++i) z += exp((double) lg[i] - mx);
        const double lz = mx + log(z);
        std::vector<int> id(n_vocab);
        for (int i = 0; i < n_vocab; ++i) id[i] = i;
        const int k = std::min(8, n_vocab);
        std::partial_sort(id.begin(), id.begin() + k, id.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
        char b[64];
        out += " [";
        for (int i = 0; i < k; ++i) { snprintf(b, sizeof(b), "%s%d=%.5f", i ? "," : "", id[i], (double) lg[id[i]] - lz); out += b; }
        out += "]";
    }
    void emit(const std::string & s) const { fputs(s.c_str(), f); fputc('\n', f); fflush(f); }
};
static mv2_dump_t g_mv2;
// Original source lines 639-651.
            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
// Original source lines 4475-4506.
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            if (g_mv2.want(slot.stats.n_gen)) {
                // no-spec: top-8 of every output row of this slot in the batch (1 row, or 2 for forced-prefix requests)
                std::string ln = "N slot=" + std::to_string(slot.id) + " n_dec=" + std::to_string(slot.stats.n_gen);
                int nr = 0;
                std::string rows;
                for (int i = 0; i < (int) batch.view.size(); ++i) {
                    if (batch.view.tokens[i].output && batch.view.tokens[i].seq_id == slot.id) { g_mv2.row(rows, slot.ctx_tgt, i); nr++; }
                }
                if (nr == 0) g_mv2.row(rows, slot.ctx_tgt, tok_idx), nr = 1;
                ln += " rows=" + std::to_string(nr) + rows + " ids=" + std::to_string(id);
                g_mv2.emit(ln);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;
// Original source lines 4564-4597.
                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                const bool mv2_on = g_mv2.want(slot.stats.n_gen);
                const std::vector<int32_t> mv2_idx = mv2_on ? slot.spec_i_batch : std::vector<int32_t>();
                std::vector<llama_token> accepted;
                mv1_scope _mv1_sa(3);
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                }
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                if (mv2_on) {
                    std::string ln = "V slot=" + std::to_string(slot.id) + " n_dec=" + std::to_string(slot.stats.n_gen) +
                        " draft=" + std::to_string(slot.spec_draft[0]) + " replay=" + std::to_string((int) slot.spec_is_replay) + " rows=" + std::to_string(mv2_idx.size());
                    for (size_t r = 0; r < mv2_idx.size(); ++r) g_mv2.row(ln, slot.ctx_tgt, mv2_idx[r]);
                    ln += " acc_n=" + std::to_string(accepted.size() - 1) + " ids=";
                    for (size_t r = 0; r < accepted.size(); ++r) ln += (r ? "," : "") + std::to_string(accepted[r]);
                    g_mv2.emit(ln);
// Original source lines 4687-4700.

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

