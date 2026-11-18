#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {
constexpr int hidden = 2048;
constexpr int experts = 256;
constexpr int ranks = 8;
constexpr int middle = 512;
constexpr int width = 3;
constexpr int graph_size = 256;
constexpr int reps = 96;
using Ids = std::array<int32_t, ranks * width>;
using Output = std::array<float, hidden * width>;

void require(bool pass, const char * why) {
    if (!pass) {
        std::fprintf(stderr, "N104 ERROR %s\n", why);
        std::exit(1);
    }
}

struct Rng {
    uint32_t seed;
    uint32_t next() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }
    float signed_unit() { return static_cast<int32_t>(next() & 0xffff) * (2.0f / 65535.0f) - 1.0f; }
};

struct WeightData {
    std::vector<float> router = std::vector<float>(hidden * experts);
    std::array<float, hidden> norm;
    std::array<float, hidden> shared_router;

    WeightData() {
        Rng rng{0x567e12ab};
        for (float & w : router) { w = .022f * rng.signed_unit(); }
        for (float & w : norm) { w = 1.0f + .08f * rng.signed_unit(); }
        for (float & w : shared_router) { w = .019f * rng.signed_unit(); }
    }
};

struct Fixture {
    const char * name;
    Output input;
    Ids ids{};
    std::array<int, 3> pair_overlap{};
    double shortlist_margin = 1.0;

    Fixture(const char * label, const WeightData & weights, float mix) : name(label) {
        Rng rng{0x25ed4117};
        for (float & x : input) { x = rng.signed_unit(); }
        for (int row = 1; row < width; ++row) {
            for (int i = 0; i < hidden; ++i) {
                float & x = input[row * hidden + i];
                x = mix * x + (1.0f - mix) * input[i];
            }
        }
        for (int row = 0; row < width; ++row) {
            std::array<double, experts> logits{};
            std::array<int, experts> order;
            std::iota(order.begin(), order.end(), 0);
            for (int e = 0; e < experts; ++e) {
                for (int i = 0; i < hidden; ++i) {
                    logits[e] += double(input[row * hidden + i]) * weights.norm[i] * weights.router[e * hidden + i];
                }
            }
            std::partial_sort(order.begin(), order.begin() + ranks + 1, order.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });
            std::copy_n(order.begin(), ranks, ids.begin() + row * ranks);
            for (int k = 0; k < ranks; ++k) {
                shortlist_margin = std::min(shortlist_margin, logits[order[k]] - logits[order[k + 1]]);
            }
        }
        for (int a = 0, p = 0; a < width; ++a) {
            for (int b = a + 1; b < width; ++b, ++p) {
                const auto b_begin = ids.begin() + b * ranks;
                for (int k = 0; k < ranks; ++k) {
                    pair_overlap[p] += std::find(b_begin, b_begin + ranks, ids[a * ranks + k]) != b_begin + ranks;
                }
            }
        }
    }
    void report() const {
        require(shortlist_margin > 1e-5, "router fixture has a near tie");
        std::fprintf(stderr, "N104 FIXTURE %s overlap=%d,%d,%d min_margin=%.8g\n",
                     name, pair_overlap[0], pair_overlap[1], pair_overlap[2], shortlist_margin);
    }
};

ggml_context * context() {
    ggml_context * ctx = ggml_init({graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true});
    require(ctx, "graph context allocation");
    return ctx;
}

void fill_q4(ggml_tensor * tensor, uint32_t seed) {
    require(tensor->type == GGML_TYPE_Q4_0 && ggml_type_size(tensor->type) == 18, "Q4_0 ABI");
    std::vector<uint8_t> tile(tensor->nb[2]);
    Rng rng{seed};
    for (int e = 0; e < tensor->ne[2]; ++e) {
        for (size_t offset = 0; offset < tile.size(); offset += 18) {
            const auto scale = ggml_fp32_to_fp16(.008f + .002f * rng.signed_unit());
            std::memcpy(tile.data() + offset, &scale, sizeof(scale));
            for (int k = 2; k < 18; k += 4) {
                const uint32_t bits = rng.next();
                std::memcpy(tile.data() + offset + k, &bits, sizeof(bits));
            }
        }
        ggml_backend_tensor_set(tensor, tile.data(), size_t(e) * tile.size(), tile.size());
    }
}

struct Weights {
    ggml_context * ctx = context();
    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, experts);
    ggml_tensor * shared_router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, hidden, middle, experts);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, hidden, middle, experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, middle, hidden, experts);
    ggml_tensor * shared_up = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hidden, middle);
    ggml_tensor * shared_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hidden, middle);
    ggml_tensor * shared_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, middle, hidden);
    ggml_backend_buffer_t buffer = nullptr;

    void load(ggml_backend_t backend, const WeightData & data) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "weight buffer allocation");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(norm, data.norm.data(), 0, ggml_nbytes(norm));
        ggml_backend_tensor_set(router, data.router.data(), 0, ggml_nbytes(router));
        ggml_backend_tensor_set(shared_router, data.shared_router.data(), 0, ggml_nbytes(shared_router));
        uint32_t seed = 0x1548ad01;
        for (ggml_tensor * t : {up, gate, down, shared_up, shared_gate, shared_down}) {
            fill_q4(t, ++seed);
        }
    }
    ~Weights() {
        if (buffer) { ggml_backend_buffer_free(buffer); }
        ggml_free(ctx);
    }
};

enum class Mode { generic, split, router };

struct Graph {
    ggml_context * ctx = context();
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_tensor * output = nullptr;
    ggml_gallocr_t alloc = nullptr;

    ggml_tensor * select(ggml_tensor * probs) {
        return ggml_argsort_top_k(ctx, probs, ranks);
    }

    ggml_tensor * block(const Weights & w, ggml_tensor * h, const int nt) {
        auto * xn = ggml_mul(ctx, ggml_rms_norm(ctx, h, 1e-6f), w.norm);
        auto * probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, w.router, xn));
        auto * ids = select(probs);
        auto * p = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, experts, nt), ids);
        p = ggml_reshape_2d(ctx, p, ranks, nt);
        auto * norm = ggml_clamp(ctx, ggml_sum_rows(ctx, p), 6.103515625e-5f, INFINITY);
        p = ggml_reshape_3d(ctx, ggml_div(ctx, p, norm), 1, ranks, nt);
        auto * x = ggml_reshape_3d(ctx, xn, hidden, 1, nt);
        auto * up = ggml_mul_mat_id(ctx, w.up, x, ids);
        auto * gate = ggml_mul_mat_id(ctx, w.gate, x, ids);
        auto * down = ggml_mul_mat_id(ctx, w.down, ggml_swiglu_split(ctx, gate, up), ids);
        auto * weighted = ggml_mul(ctx, down, p);
        ggml_tensor * routed = nullptr;
        for (int r = 0; r < ranks; ++r) {
            auto * slice = ggml_view_2d(ctx, weighted, hidden, nt, weighted->nb[2], r * weighted->nb[1]);
            routed = routed ? ggml_add(ctx, routed, slice) : slice;
        }
        auto * su = ggml_mul_mat(ctx, w.shared_up, xn);
        auto * sg = ggml_mul_mat(ctx, w.shared_gate, xn);
        auto * sd = ggml_mul_mat(ctx, w.shared_down, ggml_swiglu_split(ctx, sg, su));
        auto * sw = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w.shared_router, xn));
        return ggml_add(ctx, ggml_add(ctx, routed, ggml_mul(ctx, sd, sw)), h);
    }

    Graph(const Weights & weights, Mode mode) {
        ggml_set_input(input);
        if (mode == Mode::router) {
            auto * xn = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-6f), weights.norm);
            output = select(ggml_soft_max(ctx, ggml_mul_mat(ctx, weights.router, xn)));
        } else if (mode == Mode::split) {
            for (int row = 0; row < width; ++row) {
                auto * h = ggml_view_2d(ctx, input, hidden, 1, input->nb[1], row * input->nb[1]);
                auto * out = block(weights, h, 1);
                ggml_format_name(out, "n104_v3_row_%d", row);
                output = output ? ggml_concat(ctx, output, out, 1) : out;
            }
        } else {
            output = block(weights, input, width);
            ggml_set_name(output, "n104_generic_q3");
        }
        ggml_build_forward_expand(graph, output);
        require(ggml_nelements(output) == (mode == Mode::router ? ranks * width : hidden * width), "graph output shape");
        int projections = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_MUL_MAT_ID) {
                ++projections;
                require(t->src[2]->ne[0] == ranks && t->src[2]->ne[1] == (mode == Mode::split ? 1 : width), "expert rank/width shape");
            }
        }
        require(projections == (mode == Mode::router ? 0 : mode == Mode::generic ? 3 : 9), "graph projection count");
    }

    void allocate(ggml_backend_t backend) {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(alloc && ggml_gallocr_alloc_graph(alloc, graph), "graph buffer allocation");
    }
    void use(const Fixture & fixture) {
        ggml_backend_tensor_set(input, fixture.input.data(), 0, sizeof(fixture.input));
    }
    void run(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CUDA graph execution");
    }
    ~Graph() {
        if (alloc) { ggml_gallocr_free(alloc); }
        ggml_free(ctx);
    }
};

void check_mappings() {
    std::ifstream maps("/proc/self/maps");
    require(maps.good(), "runtime mapping inventory unavailable");
    const std::array<std::string, 3> expected = {
        "libggml.so.0.26.0", "libggml-base.so.0.26.0", "libggml-cuda.so"};
    std::array<bool, 3> found{};
    for (std::string line; std::getline(maps, line); ) {
        const auto pos = line.find('/');
        if (pos == std::string::npos) { continue; }
        const auto path = line.substr(pos);
        for (size_t i = 0; i < expected.size(); ++i) {
            if (line.find("/" + expected[i]) == std::string::npos) { continue; }
            require(path == "/runtime/" + expected[i], "unexpected ggml library mapping");
            found[i] = true;
        }
    }
    require(std::all_of(found.begin(), found.end(), [](bool value) { return value; }), "missing ggml library mapping");
    for (const auto & lib : expected) { std::fprintf(stderr, "N104 MAP /runtime/%s\n", lib.c_str()); }
}

void compare(ggml_backend_t cuda, Graph & generic, Graph & split, Graph & router, const Fixture & f) {
    router.use(f);
    router.run(cuda);
    Ids ids{};
    for (int row = 0; row < width; ++row) {
        ggml_backend_tensor_get(router.output, ids.data() + row * ranks,
                                row * router.output->nb[1], ranks * sizeof(int32_t));
    }
    require(ids == f.ids, "device router IDs differ from seeded reference");
    std::fprintf(stderr, "N104 ROUTER %s ids=24/24\n", f.name);
    Output base{}, candidate{};
    generic.use(f);
    std::fprintf(stderr, "N104 GENERIC_BEGIN %s\n", f.name);
    generic.run(cuda);
    std::fprintf(stderr, "N104 GENERIC_END %s\n", f.name);
    ggml_backend_tensor_get(generic.output, base.data(), 0, sizeof(base));
    split.use(f);
    std::fprintf(stderr, "N104 SPLIT_BEGIN %s\n", f.name);
    split.run(cuda);
    std::fprintf(stderr, "N104 SPLIT_END %s\n", f.name);
    ggml_backend_tensor_get(split.output, candidate.data(), 0, sizeof(candidate));
    for (int row = 0; row < width; ++row) {
        int mismatches = 0;
        double delta = 0.0, magnitude = 0.0, max_abs = 0.0;
        for (int i = row * hidden; i < (row + 1) * hidden; ++i) {
            require(std::isfinite(base[i]) && std::isfinite(candidate[i]), "nonfinite output");
            const double diff = double(base[i]) - candidate[i];
            mismatches += std::memcmp(&base[i], &candidate[i], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::abs(diff));
            delta += diff * diff;
            magnitude += double(base[i]) * base[i];
        }
        require(magnitude > 0, "degenerate output");
        std::fprintf(stderr, "N104 NUMERIC %s row=%d finite=2048 mismatches=%d max_abs=%.9g relative_rms=%.9g\n",
                     f.name, row, mismatches, max_abs, std::sqrt(delta / magnitude));
    }
}

double measure(ggml_backend_t cuda, Graph & graph) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) { graph.run(cuda); }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps;
}

void perf(ggml_backend_t cuda, Graph & generic, Graph & split, const Fixture & f) {
    generic.use(f);
    split.use(f);
    for (int i = 0; i < 8; ++i) {
        generic.run(cuda);
        split.run(cuda);
    }
    const double g1 = measure(cuda, generic);
    const double s1 = measure(cuda, split);
    const double s2 = measure(cuda, split);
    const double g2 = measure(cuda, generic);
    std::fprintf(stderr, "N104 ABBA %s repeats=%d generic1_us=%.6f split1_us=%.6f split2_us=%.6f generic2_us=%.6f\n",
                 f.name, reps, g1, s1, s2, g2);
}
} // namespace

int main(int argc, char ** argv) {
    require(argc == 2, "expected --fixture, --check, or --perf");
    const bool offline = std::strcmp(argv[1], "--fixture") == 0;
    const bool diagnostic = std::strcmp(argv[1], "--check") == 0;
    require(offline || diagnostic || std::strcmp(argv[1], "--perf") == 0, "unexpected test mode");
    const WeightData data;
    const std::array<Fixture, 2> fixtures = {{{"independent", data, 1.0f}, {"partial", data, .34f}}};
    const auto & independent = fixtures[0].pair_overlap;
    const auto & partial = fixtures[1].pair_overlap;
    require(*std::max_element(independent.begin(), independent.end()) <= 2, "independent rows overlap too much");
    require(*std::min_element(partial.begin(), partial.end()) >= 2 &&
            *std::max_element(partial.begin(), partial.end()) <= 7, "partial rows outside realistic overlap");
    for (const auto & f : fixtures) { f.report(); }
    ggml_backend_t cuda = nullptr;
    {
        Weights weights;
        Graph generic(weights, Mode::generic), split(weights, Mode::split), router(weights, Mode::router);
        if (offline) { return 0; }
        ggml_backend_reg_t reg = ggml_backend_load("/runtime/libggml-cuda.so");
        require(reg && ggml_backend_reg_dev_count(reg) == 1, "expected one CUDA device");
        cuda = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        require(cuda && std::strcmp(ggml_backend_name(cuda), "CUDA0") == 0, "expected CUDA0 backend");
        check_mappings();
        weights.load(cuda, data);
        generic.allocate(cuda);
        split.allocate(cuda);
        router.allocate(cuda);
        for (const auto & f : fixtures) {
            if (diagnostic) { compare(cuda, generic, split, router, f); }
            else { perf(cuda, generic, split, f); }
        }
    }
    ggml_backend_free(cuda);
}
