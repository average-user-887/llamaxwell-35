#!/usr/bin/env python3
from pathlib import Path
import difflib

here = Path(__file__).resolve().parent
source = here.parent/'night-one-graph-draft/src/src'


def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def emit(path, before, after):
    (here / path.name).write_text(after)
    (here / (path.stem + path.suffix + '.patch')).write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), f'a/src/{path.name}', f'b/src/{path.name}')))

path = source/'llama-context.h'
before = text = path.read_text()
text = change(text, '#pragma once\n', '#pragma once\n\n#include "compact-readback.h"\n')
text = change(text, '    int decode_mtp(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query);',
    '''    int decode_mtp(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query);
    int decode_mtp_output(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query,
            const occamy_compact::Mask * mask, occamy_compact::Result * compact);''')
emit(path, before, text)

path = source/'llama-context.cpp'
before = text = path.read_text()
text = change(text, '#include <cstring>\n', '#include <cstring>\n#include <dlfcn.h>\n')
text = change(text, '''int llama_context::decode_mtp(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query) {
    if (!memory) {''', '''int llama_context::decode_mtp(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query) {
    return decode_mtp_output(batch_inp, mtp_cache_catch_up, last_query, nullptr, nullptr);
}

// Resolve the extra operator in the exact DSO that owns this CUDA backend,
// including the usual RTLD_LOCAL dynamic backend loading mode.
static bool occamy_read_compact(ggml_backend_t backend, const ggml_tensor * tensor,
        const occamy_compact::Mask * mask, void * pinned, occamy_compact::Result * output) {
    if (!backend || strncmp(ggml_backend_name(backend), "CUDA", 4) != 0) {
        return false;
    }
    using compact_fn = bool (*)(ggml_backend_t, const ggml_tensor *, const occamy_compact::Mask *,
            void *, occamy_compact::Result *);
    void * anchor = mv3_proc<void *>(backend, "ggml_backend_cuda_mv3_group_begin");
    Dl_info info{};
    if (!anchor || !dladdr(anchor, &info) || !info.dli_fname) {
        return false;
    }
    void * owner = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    if (!owner) {
        return false;
    }
    auto fn = reinterpret_cast<compact_fn>(dlsym(owner, "ggml_backend_cuda_occamy_compact_v1"));
    const bool launched = fn && fn(backend, tensor, mask, pinned, output);
    dlclose(owner);
    return launched;
}

int llama_context::decode_mtp_output(const llama_batch_ext & batch_inp, bool mtp_cache_catch_up, bool last_query,
        const occamy_compact::Mask * compact_mask, occamy_compact::Result * compact_result) {
    if (!memory) {''')
text = change(text, '''    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();''', '''    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    const bool compact_shape = compact_result && compact_mask && !mtp_cache_catch_up && !last_query &&
            model.arch == LLM_ARCH_QWEN35MOE && model.type == LLM_TYPE_35B_A3B &&
            hparams.n_layer_nextn == 1 && hparams.n_expert_used() == 8 &&
            n_vocab == occamy_compact::vocab_size && vocab.get_suppress_tokens().empty() &&
            cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && cparams.n_seq_max == 1 &&
            cparams.causal_attn && !cparams.warmup && !cparams.training && !cparams.pipeline_parallel &&
            cparams.embeddings_nextn && !cparams.embeddings_nextn_masked &&
            cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && !output_all && !has_samplers &&
            !has_graph_adapter() && n_tokens_all == 2 && n_outputs_all == 2;''')
# Only replace the decode section, not enc/dec output sites elsewhere in the file.
start, rest = text.split('int llama_context::decode_mtp_output(', 1)
end = rest.index('\n//\n// output\n')
body, tail = rest[:end], rest[end:]
a = body.index('        // extract logits\n')
b = body.index('        // extract embeddings\n', a)
body = body[:a] + '''        const bool raw_logits = logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers);
        ggml_backend_t backend_res = raw_logits ? ggml_backend_sched_get_tensor_backend(sched.get(), t_logits) : nullptr;
        const auto read_raw_logits = [&] {
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
            GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
            const size_t bytes = n_outputs*n_vocab*sizeof(float);
            ggml_backend_tensor_get_async(backend_res, t_logits, logits.data + n_outputs_prev*n_vocab, 0, bytes);
            if (compact_result) {
                compact_result->raw += bytes;
            }
        };
        const auto & out_ids = balloc->get_out_ids();
        const bool try_compact = compact_shape && raw_logits && backend_res && buf_output &&
                ggml_backend_buffer_get_type(buf_output.get()) == ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend_res)) &&
                ubatch.n_tokens == 2 && ubatch.n_seqs_unq == 1 && ubatch.token && !ubatch.embd && !ubatch.embd_state &&
                ubatch.n_seq_id[0] == 1 && ubatch.n_seq_id[1] == 1 &&
                ubatch.seq_id[0][0] == 0 && ubatch.seq_id[1][0] == 0 &&
                ubatch.token[0] == batch_inp.tokens[0].id && ubatch.token[1] == batch_inp.tokens[1].id &&
                n_outputs_prev == 0 && n_outputs == 2 && out_ids.size() == 2 && out_ids[0] == 0 && out_ids[1] == 1;
        if (raw_logits && !try_compact) {
            read_raw_logits();
        }

''' + body[b:]
body = change(body, '        // extract nextn embeddings before\n', '        ggml_backend_t queued_hidden_backend = nullptr;\n        // extract nextn embeddings before\n')
body = change(body, '''                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs''', '''                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                queued_hidden_backend = backend_h;
                extract_all_idxs''')
body = change(body, '        if (has_samplers) {\n            const auto stride', '''        if (try_compact) {
            // The preceding hidden-state copy also flushes queued CUDA work on the
            // logits stream. Keep this graph's original F32 output live for fallback.
            const bool launched = queued_hidden_backend == backend_res &&
                    occamy_read_compact(backend_res, t_logits, compact_mask, logits.data, compact_result);
            compact_result->launched = launched;
            compact_result->compact = launched ? 2 * sizeof(occamy_compact::Choice) : 0;
            compact_result->compact_only = launched && !compact_result->shadow && occamy_compact::valid(compact_result->rows);
            if (!compact_result->compact_only) {
                read_raw_logits();
            }
        }

        if (has_samplers) {
            const auto stride''')
text = start + 'int llama_context::decode_mtp_output(' + body + tail
text = change(text, '''void llama_mtp_enable_last_query(llama_context * ctx) {''', '''int32_t llama_occamy_compact_decode(llama_context * ctx, llama_batch_ext * batch,
        const occamy_compact::Mask & mask, bool shadow, occamy_compact::Result & result) {
    result = {};
    result.requested = true;
    result.shadow = shadow;
    const auto & vocab = ctx->get_model().vocab;
    const bool admitted = std::is_sorted(mask.begin(), mask.end()) &&
            std::adjacent_find(mask.begin(), mask.end()) == mask.end() &&
            std::all_of(mask.begin(), mask.end(), [&](llama_token id) {
                return id >= 0 && id < vocab.n_tokens() && vocab.is_eog(id);
            });
    return ctx->decode_mtp_output(*batch, false, false, admitted ? &mask : nullptr, &result);
}

void llama_mtp_enable_last_query(llama_context * ctx) {''')
emit(path, before, text)
