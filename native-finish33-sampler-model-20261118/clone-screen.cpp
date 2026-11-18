#include "sampler-layout.inc"

#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <memory>

using Clock = std::chrono::steady_clock;
constexpr size_t vocab = 248320;
constexpr int samples = 600;

static common_sampler_ptr make_sampler(bool full) {
    common_params_sampling p;
    p.temp = 0;
    p.seed = 7;
    auto * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(vocab, p.penalty_last_n, p.penalty_repeat, p.penalty_freq, p.penalty_present));
    llama_sampler_chain_add(chain, llama_sampler_init_dry(nullptr, p.dry_multiplier, p.dry_base, p.dry_allowed_length, p.dry_penalty_last_n, nullptr, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(p.top_n_sigma));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_typical(p.typ_p, p.min_keep));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, p.min_keep));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(p.min_p, p.min_keep));
    llama_sampler_chain_add(chain, llama_sampler_init_xtc(p.xtc_probability, p.xtc_threshold, p.min_keep, p.seed));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(p.temp, p.dynatemp_range, p.dynatemp_exponent));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    common_sampler_ptr sampler(new common_sampler {
        p, nullptr, nullptr, chain, ring_buffer<llama_token>(p.n_prev), {}, {}, std::mt19937(p.seed),
    });
    if (full) {
        sampler->cur.resize(vocab);
        for (size_t i = 0; i < vocab; ++i) {
            sampler->cur[i] = {static_cast<llama_token>(i), static_cast<float>(i % 211) / 16, 0};
        }
    }
    for (int i = 0; i < p.n_prev; ++i) {
        common_sampler_accept(sampler.get(), i, true);
    }
    sampler->cur_p = {sampler->cur.data(), sampler->cur.size(), -1, false};
    return sampler;
}

static void measure(const char * name, common_sampler * sampler, const std::vector<float> & evict) {
    std::vector<double> clone_time, free_time, duration;
    clone_time.reserve(samples);
    free_time.reserve(samples);
    duration.reserve(samples);
    volatile float used = 0;
    for (int i = -60; i < samples; ++i) {
        for (size_t j = 0; j < evict.size(); j += 16) {
            used = used + evict[j];
        }
        const auto start = Clock::now();
        common_sampler * clone = common_sampler_clone(sampler);
        const auto saved = Clock::now();
        common_sampler_free(clone);
        const auto end = Clock::now();
        if (i >= 0) {
            clone_time.push_back(std::chrono::duration<double, std::micro>(saved - start).count());
            free_time.push_back(std::chrono::duration<double, std::micro>(end - saved).count());
            duration.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
    }
    const auto results = [](std::vector<double> & data) {
        std::sort(data.begin(), data.end());
        double sum = 0;
        for (double value : data) {
            sum += value;
        }
        std::printf(" mean_us=%.3f p50_us=%.3f p90_us=%.3f", sum / samples, data[samples / 2], data[9 * samples / 10]);
    };
    std::printf("clone_screen %s samples=%d token_bytes=%zu clone", name, samples, sampler->cur.size() * sizeof(llama_token_data));
    results(clone_time);
    std::printf(" free");
    results(free_time);
    std::printf(" end_to_end");
    results(duration);
    std::printf("\n");
}

int main() {
    Dl_info actual = {};
    if (!dladdr(reinterpret_cast<const void *>(common_sampler_clone), &actual)) {
        return 1;
    }
    std::printf("linked_common=%s vocab=%zu token_data_bytes=%zu\n", actual.dli_fname, vocab, sizeof(llama_token_data));
    auto instrument = make_sampler(true);
    auto empty = make_sampler(false);
    measure("empty", empty.get(), {});
    measure("full_warm", instrument.get(), {});
    const std::vector<float> evict(64 * 1024 * 1024 / sizeof(float), 1);
    measure("full_after_64MiB_sweep", instrument.get(), evict);
}
