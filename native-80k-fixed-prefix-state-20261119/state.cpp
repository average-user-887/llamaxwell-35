#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int context_size = 84224;
constexpr int prompt_size = 80000;
constexpr int prefix_tokens = 80167;
constexpr int vocab_size = 248320;
constexpr int kv_padded = 80384;
constexpr size_t recurrent_size = 65864420; // exclude sequence magic and ID
constexpr uint32_t conv_size = 98304, ssm_size = 2097152;
constexpr size_t max_state = 1700ull * 1024 * 1024;
using Bytes = std::vector<uint8_t>;
using Scores = std::vector<float>;
using Tokens = std::vector<llama_token>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Shape {
    int calls = 0, input = 0, output = 0, sparse = 0, invalid_sparse = 0;
};
thread_local Shape * observed = nullptr;
std::atomic<bool> offloaded{false};

using Submit = decltype(&ggml_backend_sched_graph_compute_async);
Submit submit() {
    static const auto fn = reinterpret_cast<Submit>(dlsym(RTLD_NEXT, "ggml_backend_sched_graph_compute_async"));
    return fn;
}

void log_callback(ggml_log_level, const char * line, void *) {
    if (std::strstr(line, "offloaded 42/42 layers to GPU")) offloaded.store(true);
    std::fputs(line, stderr);
}
}

extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    if (observed) {
        const auto * input = ggml_graph_get_tensor(graph, "inp_tokens");
        const auto * output = ggml_graph_get_tensor(graph, "result_output");
        ++observed->calls;
        observed->input = input ? int(input->ne[0]) : -1;
        observed->output = output ? int(output->ne[1]) : -1;
    }
    const auto fn = submit();
    return fn ? fn(sched, graph) : GGML_STATUS_FAILED;
}

#include "dispatch.inc"

void ggml_cuda_occamy_fa_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const auto fn = cuda_symbol<decltype(&ggml_cuda_occamy_fa_sparse)>(
        "_Z26ggml_cuda_occamy_fa_sparseR25ggml_backend_cuda_contextP11ggml_tensor");
    fn(ctx, dst);
    if (observed) {
        ++observed->sparse;
        const auto * q = dst->src[0];
        const auto * k = dst->src[1];
        if (!q || !k || q->ne[1] != observed->input || k->ne[1] != kv_padded || k->ne[2] != 2) {
            ++observed->invalid_sparse;
        }
    }
}

namespace {
struct Reader {
    const Bytes & data;
    size_t at = 0;

    const uint8_t * take(size_t size) {
        require(at <= data.size() && size <= data.size() - at, "truncated state envelope");
        const auto * value = data.data() + at;
        at += size;
        return value;
    }

    template <class T> T read() {
        T value{};
        std::memcpy(&value, take(sizeof(T)), sizeof(T));
        return value;
    }
};

struct State {
    Bytes kv_tail, conv, ssm;
    bool kv_prefix_equal = true;
};

// N40 single-stream, F16, nontransposed Occamy KV and active recurrent format.
State extract(const Bytes & bytes, int count, const Bytes * prefix = nullptr, int base = prefix_tokens) {
    const auto metadata_width = [&] {
        const size_t fixed = 32 + 20 * (12 + size_t(count) * 1024) + recurrent_size;
        require(bytes.size() >= fixed && (bytes.size() - fixed) % count == 0, "KV envelope dimensions");
        const size_t width = (bytes.size() - fixed) / count;
        require(width == 12 || width == 24, "unexpected KV cell metadata width");
        return width;
    }();
    Reader r{bytes};
    require(r.read<uint32_t>() == 0xaf143cd8 && r.read<uint32_t>() == 0 &&
            r.read<uint32_t>() == 1 && r.read<uint32_t>() == uint32_t(count), "KV envelope magic/count");
    const auto * metadata = r.take(metadata_width * count);
    for (int i = 0; i < count; ++i) {
        uint32_t pos{}, sequences{}, sequence{};
        const auto * cell = metadata + metadata_width * i;
        std::memcpy(&pos, cell, 4);
        std::memcpy(&sequences, cell + 4, 4);
        std::memcpy(&sequence, cell + metadata_width - 4, 4);
        require(pos == uint32_t(i) && sequences == 1 && sequence == 0, "KV cell layout is not contiguous sequence zero");
    }
    require(r.read<uint32_t>() == 0 && r.read<uint32_t>() == 10 &&
            r.read<uint32_t>() == 0 && r.read<uint32_t>() == 0, "unexpected F16 KV layer layout");

    State state;
    Reader reference{prefix ? *prefix : bytes};
    if (prefix) {
        require(count >= base, "stage is shorter than the baseline");
        reference.take(16);
        const auto * prefix_metadata = reference.take(metadata_width * base);
        state.kv_prefix_equal = std::memcmp(metadata, prefix_metadata, metadata_width * base) == 0;
        reference.take(16);
        state.kv_tail.insert(state.kv_tail.end(), metadata + metadata_width * base, metadata + metadata_width * count);
    }
    for (int i = 0; i < 20; ++i) {
        require(r.read<uint32_t>() == GGML_TYPE_F16 && r.read<uint64_t>() == 1024, "non-F16 KV or wrong row stride");
        const auto * row = r.take(size_t(count) * 1024);
        if (prefix) {
            reference.take(12);
            const auto * old = reference.take(size_t(base) * 1024);
            state.kv_prefix_equal &= std::memcmp(row, old, size_t(base) * 1024) == 0;
            state.kv_tail.insert(state.kv_tail.end(), row + size_t(base) * 1024, row + size_t(count) * 1024);
        }
    }
    require(r.read<uint32_t>() == 1 && r.read<uint32_t>() == uint32_t(count - 1) &&
            r.read<uint32_t>() == 0 && r.read<uint32_t>() == 0 && r.read<uint32_t>() == 40, "active recurrent metadata");
    for (auto [size, destination] : {std::pair{conv_size, &state.conv}, std::pair{ssm_size, &state.ssm}}) {
        destination->reserve(30ull * size);
        for (int layer = 0; layer < 30; ++layer) {
            require(r.read<uint32_t>() == GGML_TYPE_F32 && r.read<uint64_t>() == size, "recurrent type or row stride");
            const auto * layer_data = r.take(size);
            destination->insert(destination->end(), layer_data, layer_data + size);
        }
    }
    require(r.at == bytes.size(), "unexpected state trailer");
    return state;
}

void snapshot(llama_context * ctx, Bytes & buffer, uint32_t flags = 0) {
    llama_synchronize(ctx);
    const auto size = llama_state_seq_get_size_ext(ctx, 0, flags);
    require(size > 0 && size <= max_state, "state is outside host RAM envelope");
    buffer.resize(size);
    require(llama_state_seq_get_data_ext(ctx, buffer.data(), size, 0, flags) == size, "state snapshot incomplete");
}

void compare_state(const char * comparison, const State & a, const State & b) {
    const bool kv = a.kv_prefix_equal && b.kv_prefix_equal && a.kv_tail == b.kv_tail;
    std::cout << "STATE " << comparison << " kv=" << kv << " conv=" << (a.conv == b.conv)
              << " ssm=" << (a.ssm == b.ssm) << " a_prefix=" << a.kv_prefix_equal
              << " b_prefix=" << b.kv_prefix_equal << '\n';
}

struct Decision {
    int raw_top, greedy;
    double gap, normalizer;
};

Decision decision(const Scores & s, const llama_vocab * vocab) {
    float top = -INFINITY, second = -INFINITY, best = -INFINITY;
    int top_id = -1, chosen = -1;
    for (int i = 0; i < int(s.size()); ++i) {
        require(std::isfinite(s[i]), "nonfinite target logit");
        if (s[i] > top) {
            second = top;
            top = s[i];
            top_id = i;
        } else if (s[i] > second) {
            second = s[i];
        }
        if (!llama_vocab_is_eog(vocab, i) && s[i] > best) {
            best = s[i];
            chosen = i;
        }
    }
    double sum = 0;
    for (const auto value : s) sum += std::exp(double(value) - top);
    require(chosen >= 0 && std::isfinite(second) && std::isfinite(sum), "invalid target score row");
    return {top_id, chosen, double(top) - second, top + std::log(sum)};
}

const char * label(int token, int serial, int native) {
    if (token == serial) return "serial";
    if (token == native) return "native";
    return "other";
}

void describe(const char * stage, const Scores & s, const Decision & d, int serial, int native) {
    std::cout << std::setprecision(9) << "SCORE " << stage << " raw=" << label(d.raw_top, serial, native)
              << " greedy=" << label(d.greedy, serial, native) << " top2_gap=" << d.gap
              << " serial_logp=" << (double(s[serial]) - d.normalizer) << " native_logp="
              << (double(s[native]) - d.normalizer) << " serial_minus_native="
              << (double(s[serial]) - s[native]) << '\n';
}

bool compare_scores(const char * comparison, const Scores & a, const Scores & b) {
    require(a.size() == b.size() && !a.empty(), "score shape mismatch");
    size_t different = 0;
    double max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite compared score");
        different += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        max_abs = std::max(max_abs, std::abs(double(a[i]) - b[i]));
    }
    std::cout << "LOGITS " << comparison << " different_f32_words=" << different << " max_abs=" << max_abs << '\n';
    return different == 0;
}

class Batch {
    llama_batch b_ = llama_batch_init(2048, 0, 1);
public:
    ~Batch() { llama_batch_free(b_); }

    void decode(llama_context * ctx, const llama_token * ids, int start, int n, bool outputs, bool last_only = false) {
        require(n > 0 && n <= 2048, "unexpected logical batch size");
        b_.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b_.token[i] = ids[i];
            b_.pos[i] = start + i;
            b_.n_seq_id[i] = 1;
            b_.seq_id[i][0] = 0;
            b_.logits[i] = outputs || (last_only && i == n - 1);
        }
        require(llama_decode(ctx, b_) == 0, "target decode failed");
    }
};

using Counter = unsigned long long (*)(unsigned);
std::array<unsigned long long, 4> counts(Counter counter) {
    return {counter(0), counter(1), counter(2), counter(3)};
}

std::vector<Scores> observed_decode(llama_context * ctx, Batch & batch, const char * stage, int start,
                                    const Tokens & ids, Counter counter) {
    const int width = int(ids.size());
    Shape shape;
    const auto old = counts(counter);
    const auto t2_old = t2_launched.load(), tries_old = t2_attempts.load();
    const auto stock_old = stock.load(), stock_cache_old = stock_cache.load();
    observed = &shape;
    count_q2.store(width == 2);
    try {
        batch.decode(ctx, ids.data(), start, width, true);
        llama_synchronize(ctx);
    } catch (...) {
        observed = nullptr;
        count_q2.store(false);
        throw;
    }
    observed = nullptr;
    count_q2.store(false);
    auto delta = counts(counter);
    for (size_t i = 0; i < delta.size(); ++i) delta[i] -= old[i];
    const auto launches = t2_launched.load() - t2_old;
    const auto attempts = t2_attempts.load() - tries_old;
    const auto fallbacks = stock.load() - stock_old + stock_cache.load() - stock_cache_old;
    std::cout << "GRAPH " << stage << " input=" << shape.input << " output=" << shape.output
              << " calls=" << shape.calls << " position=" << start << " sparse=" << shape.sparse
              << " invalid_sparse=" << shape.invalid_sparse << " padded_kv=" << kv_padded
              << " t2=" << launches << " t2_attempts=" << attempts << " t2_fallback=" << fallbacks
              << " moe=" << delta[0] << ',' << delta[1] << ',' << delta[2] << ',' << delta[3] << '\n';
    require(shape.calls == 1 && shape.input == width && shape.output == width &&
            shape.sparse == 10 && shape.invalid_sparse == 0 &&
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == start + width - 1,
            "actual target graph, long sparse branch or position differs");
    if (width == 2) {
        require(launches == 30 && attempts == 30 && fallbacks == 0 && wrong_shape.load() == 0 &&
                delta == std::array<unsigned long long, 4>{0, 0, 40, 40}, "actual Q2 GDN or full MoE cohort differs");
    } else {
        require(launches == 0 && delta == std::array<unsigned long long, 4>{}, "Q1 unexpectedly used Q2 cohort");
    }

    std::vector<Scores> output;
    for (int row = 0; row < width; ++row) {
        const auto * scores = llama_get_logits_ith(ctx, row);
        require(scores != nullptr, "missing target output row");
        output.emplace_back(scores, scores + vocab_size);
    }
    return output;
}

Context new_context(llama_model * model) {
    auto p = llama_context_default_params();
    p.n_ctx = context_size;
    p.n_batch = 2048;
    p.n_ubatch = 1024;
    p.n_seq_max = 1;
    p.n_rs_seq = 1;
    p.n_outputs_max = p.n_outputs_max_per_seq = 2;
    p.n_threads = p.n_threads_batch = 6;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.offload_kqv = p.op_offload = true;
    p.kv_unified = false;
    p.type_k = p.type_v = GGML_TYPE_F16;
    Context ctx(llama_init_from_model(model, p), llama_free);
    require(ctx && llama_n_ctx(ctx.get()) == context_size && llama_n_batch(ctx.get()) == 2048 &&
            llama_n_ubatch(ctx.get()) == 1024 && llama_n_rs_seq(ctx.get()) == 1 &&
            llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == -1, "target context shape mismatch");
    return ctx;
}

Tokens load_tape(const char * path) {
    std::ifstream input(path);
    Tokens tape;
    llama_token token;
    while (input >> token) {
        require(token >= 0 && token < vocab_size, "private tape contains an invalid token");
        tape.push_back(token);
    }
    require(input.eof() && tape.size() == prefix_tokens + 4, "private tape length or encoding changed");
    return tape;
}

void verify_runtime_maps() {
    const std::array<const char *, 4> names{"libggml-cuda.so", "libllama.so", "libggml.so", "libggml-base.so"};
    std::set<std::string> expected, actual;
    for (auto name : names) {
        char path[PATH_MAX]{};
        require(realpath((std::string("/candidate/") + name).c_str(), path), "cannot resolve exact target runtime");
        expected.emplace(path);
    }
    std::ifstream maps("/proc/self/maps");
    require(maps.is_open(), "cannot read target process mappings");
    std::string line;
    while (std::getline(maps, line)) {
        const auto slash = line.find('/');
        if (slash == std::string::npos) continue;
        const auto path = line.substr(slash);
        const auto base = path.substr(path.find_last_of('/') + 1);
        for (auto name : names) {
            if (base.rfind(name, 0) == 0) actual.insert(path);
        }
    }
    require(expected.size() == names.size() && actual == expected, "exact target runtime mappings differ");
    for (const auto & path : actual) std::fprintf(stderr, "QUALITY80_MAPPED %s\n", path.c_str());
}

void run(const char * model_path, const Tokens & tape) {
    llama_log_set(log_callback, nullptr);
    ggml_backend_load_all_from_path("/candidate");
    llama_backend_init();
    verify_runtime_maps();
    Dl_info origin{};
    require(dladdr(reinterpret_cast<void *>(&llama_decode), &origin) && origin.dli_fname &&
            std::string(origin.dli_fname).rfind("/candidate/libllama.so", 0) == 0, "target libllama resolves outside candidate");
    const auto cuda = ggml_backend_dev_by_name("CUDA0");
    require(cuda && ggml_backend_dev_type(cuda) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            std::strstr(ggml_backend_dev_description(cuda), "M6000"), "admitted CUDA0 M6000 missing");
    ggml_backend_dev_t devices[]{cuda, nullptr};
    llama_model_kv_override overrides[2]{};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    std::strcpy(overrides[0].key, "tokenizer.ggml.pre");
    std::strcpy(overrides[0].val_str, "qwen35");
    auto p = llama_model_default_params();
    p.devices = devices;
    p.kv_overrides = overrides;
    p.n_gpu_layers = 999;
    p.progress_callback = [](float, void *) { return true; };
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path, p), llama_model_free);
    require(model && offloaded.load(), "model did not report actual 42/42 GPU offload");
    const auto * vocab = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab) == vocab_size, "unexpected target vocabulary");
    const int keep = tape[prefix_tokens], correct = tape[prefix_tokens + 1];
    const int serial = tape[prefix_tokens + 2], native = tape[prefix_tokens + 3];
    require(serial != native, "expected different archived decision170 choices");
    const auto counter = cuda_symbol<Counter>("ggml_cuda_occamy_q2_counter");

    auto ctx = new_context(model.get());
    Batch batch;
    int start = 0;
    auto prefill_to = [&](int end) {
        batch.decode(ctx.get(), tape.data() + start, start, end - start, false);
        start = end;
    };
    for (int end = 2048; end <= 77824; end += 2048) prefill_to(end);
    for (int end : {78972, 79996}) prefill_to(end);
    // The actual server requests the last prompt logit on the four-token tail.
    batch.decode(ctx.get(), tape.data() + start, start, prompt_size - start, false, true);
    for (int at = prompt_size; at < prefix_tokens; ++at) batch.decode(ctx.get(), tape.data() + at, at, 1, true);
    require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == prefix_tokens - 1,
            "teacher-forced prefix position mismatch");
    Bytes prefix, scratch, partial;
    prefix.reserve(max_state);
    scratch.reserve(max_state);
    snapshot(ctx.get(), prefix);
    snapshot(ctx.get(), partial, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    require(partial.size() == recurrent_size + 8 &&
            std::memcmp(prefix.data(), partial.data(), 8) == 0 &&
            std::memcmp(prefix.data() + prefix.size() - recurrent_size, partial.data() + 8, recurrent_size) == 0,
            "N40 full and partial state envelopes differ");
    Bytes().swap(partial);
    extract(prefix, prefix_tokens);
    std::cout << "PREFIX positions=" << prefix_tokens << " bytes=" << prefix.size() << " offload=42/42 rs=1 outputs=2\n";

    auto restore = [&](const char * stage) {
        ctx.reset();
        ctx = new_context(model.get());
        require(llama_state_seq_set_data_ext(ctx.get(), prefix.data(), prefix.size(), 0, 0) == prefix.size(),
                "fresh target prefix restore failed");
        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == prefix_tokens - 1,
                "restored target prefix position differs");
        snapshot(ctx.get(), scratch);
        require(scratch == prefix, "restored target prefix bytes differ");
        std::cout << "RESTORE " << stage << " bitwise=1\n";
    };
    auto capture = [&](int count) {
        snapshot(ctx.get(), scratch);
        return extract(scratch, count, &prefix);
    };

    restore("accepted");
    const auto accepted = observed_decode(ctx.get(), batch, "accepted", prefix_tokens, {keep, correct}, counter);
    const auto accepted_row0 = decision(accepted[0], vocab);
    const auto accepted_row1 = decision(accepted[1], vocab);
    describe("accepted_row1", accepted[1], accepted_row1, serial, native);
    std::cout << "ACCEPTED row0_matches_archived_accepted_token=" << (accepted_row0.greedy == correct) << '\n';
    const auto accepted_state = capture(prefix_tokens + 2);

    restore("serial");
    const auto serial_first = observed_decode(ctx.get(), batch, "serial_first", prefix_tokens, {keep}, counter);
    const auto serial_kept = capture(prefix_tokens + 1);
    const auto serial_second = observed_decode(ctx.get(), batch, "serial_second", prefix_tokens + 1, {correct}, counter);
    const auto serial_decision = decision(serial_second[0], vocab);
    describe("serial_row1_conditioning", serial_second[0], serial_decision, serial, native);
    const auto serial_state = capture(prefix_tokens + 2);
    compare_scores("accepted_row1_vs_serial", accepted[1], serial_second[0]);
    compare_state("accepted_vs_serial", accepted_state, serial_state);

    std::array<int, 2> wrong{};
    size_t nwrong = 0;
    for (const auto token : {42, 43, 44, 45, 46, 47}) {
        if (token != keep && token != correct && token != accepted_row0.greedy && !llama_vocab_is_eog(vocab, token)) {
            wrong[nwrong++] = token;
            if (nwrong == wrong.size()) break;
        }
    }
    require(nwrong == wrong.size(), "two distinct forced proposals unavailable");
    struct Rewind {
        Scores first, bad_row, after;
        State untrimmed, kept, corrected;
        bool genuine;
    };
    auto rewind = [&](const char * mode, const char * correction, int bad) {
        restore(mode);
        auto scores = observed_decode(ctx.get(), batch, mode, prefix_tokens, {keep, bad}, counter);
        const auto winner = decision(scores[0], vocab).greedy;
        const bool genuine = scores[0][winner] > scores[0][bad];
        auto untrimmed = capture(prefix_tokens + 2);
        auto mem = llama_get_memory(ctx.get());
        require(llama_memory_seq_rm(mem, 0, prefix_tokens + 1, -1) &&
                llama_memory_seq_pos_max(mem, 0) == prefix_tokens, "one-position native target trim failed");
        std::cout << "TRIM " << mode << " removed=1 genuine=" << genuine << " retained_position=" << prefix_tokens << '\n';
        auto kept = capture(prefix_tokens + 1);
        auto after = observed_decode(ctx.get(), batch, correction, prefix_tokens + 1, {correct}, counter);
        describe(correction, after[0], decision(after[0], vocab), serial, native);
        return Rewind{std::move(scores[0]), std::move(scores[1]), std::move(after[0]),
                      std::move(untrimmed), std::move(kept), capture(prefix_tokens + 2), genuine};
    };
    const auto a = rewind("wrong_a", "correct_a", wrong[0]);
    const auto b = rewind("wrong_b", "correct_b", wrong[1]);
    compare_state("wrong_untrimmed_a_vs_b", a.untrimmed, b.untrimmed);
    compare_state("wrong_retained_a_vs_b", a.kept, b.kept);
    compare_state("wrong_corrected_a_vs_b", a.corrected, b.corrected);
    compare_state("wrong_retained_a_vs_serial", a.kept, serial_kept);
    compare_state("wrong_corrected_a_vs_serial", a.corrected, serial_state);
    compare_scores("wrong_row0_a_vs_b", a.first, b.first);
    compare_scores("wrong_row1_a_vs_b", a.bad_row, b.bad_row);
    compare_scores("wrong_row0_a_vs_accepted_row0", a.first, accepted[0]);
    compare_scores("wrong_corrected_a_vs_b", a.after, b.after);
    compare_scores("wrong_corrected_a_vs_serial", a.after, serial_second[0]);
    std::cout << "COMPLETE width_flip_matches_archived=" << (accepted_row1.greedy == native && serial_decision.greedy == serial)
              << " genuine_two_rejections=" << (a.genuine && b.genuine) << '\n';
}

void self_test() {
    require(submit(), "scheduler forwarder missing");
    auto make = [](int n, uint8_t new_value) {
        Bytes bytes;
        auto add = [&](auto value) {
            const auto * p = reinterpret_cast<const uint8_t *>(&value);
            bytes.insert(bytes.end(), p, p + sizeof(value));
        };
        for (uint32_t v : {0xaf143cd8u, 0u, 1u, uint32_t(n)}) add(v);
        for (int i = 0; i < n; ++i) { add(uint32_t(i)); add(1u); add(0u); }
        for (uint32_t v : {0u, 10u, 0u, 0u}) add(v);
        for (int i = 0; i < 20; ++i) {
            add(uint32_t(GGML_TYPE_F16)); add(uint64_t(1024));
            bytes.resize(bytes.size() + size_t(n) * 1024);
            if (n > 1) bytes.back() = new_value;
        }
        for (uint32_t v : {1u, uint32_t(n - 1), 0u, 0u, 40u}) add(v);
        for (auto size : {conv_size, ssm_size}) {
            for (int i = 0; i < 30; ++i) {
                add(uint32_t(GGML_TYPE_F32)); add(uint64_t(size));
                bytes.resize(bytes.size() + size);
            }
        }
        return bytes;
    };
    auto prefix = make(1, 0);
    auto next = make(2, 3);
    const auto a = extract(next, 2, &prefix, 1);
    const auto b = extract(make(2, 5), 2, &prefix, 1);
    require(a.kv_prefix_equal && b.kv_prefix_equal && a.kv_tail != b.kv_tail && a.conv == b.conv
            && a.ssm == b.ssm, "exact active component comparison self-test");
    const size_t first_key = 16 + 24 + 16 + 12;
    next.at(first_key) = 1;
    require(!extract(next, 2, &prefix, 1).kv_prefix_equal, "historical KV change self-test");
    require(compare_scores("self_equal", {1, 2}, {1, 2}) &&
            !compare_scores("self_different", {1, 2}, {1, 3}), "score bit-comparison self-test");
    std::cout << "CPU_SELFTEST_PASS\n";
}
}

int main(int argc, char ** argv) {
    try {
        std::cout << std::boolalpha << std::unitbuf;
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
        } else {
            require(argc == 3, "usage: state MODEL PRIVATE_TAPE");
            run(argv[1], load_tape(argv[2]));
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "QUALITY80_STATE_ERROR %s\n", e.what());
        return 2;
    }
}
