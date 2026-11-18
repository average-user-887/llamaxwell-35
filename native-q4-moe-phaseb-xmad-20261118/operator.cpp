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
#include <dlfcn.h>
#include <unistd.h>
#include <numeric>
#include <string>
#include <vector>

namespace {
constexpr int hidden = 2048;
constexpr int experts = 256;
constexpr int ranks = 8;
constexpr int middle = 512;
constexpr int width = 4;
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
    std::array<int, width*(width-1)/2> pair_overlap{};
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
        for (int row=0;row<width;++row) {
            std::fprintf(stderr,"N104 FIXTURE_IDS %s row=%d ids=",name,row);
            for(int k=0;k<ranks;++k) std::fprintf(stderr,"%s%d",k?",":"",ids[row*ranks+k]);
            std::fprintf(stderr,"\n");
        }
        require(shortlist_margin > 1e-5, "router fixture has a near tie");
        std::fprintf(stderr, "N104 FIXTURE %s overlap=%d,%d,%d,%d,%d,%d min_margin=%.8g\n",
                     name, pair_overlap[0], pair_overlap[1], pair_overlap[2], pair_overlap[3], pair_overlap[4], pair_overlap[5], shortlist_margin);
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

enum class Mode { generic, split, router, phase };

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
        // The same seeded rows are evaluated repeatedly during timing.
        // Pin their allocation so GGML cannot reuse it for the residual.
        ggml_set_output(input);
        if (mode == Mode::phase) {
            auto * xn=ggml_mul(ctx,ggml_rms_norm(ctx,input,1e-6f),weights.norm);
            auto * ids=select(ggml_soft_max(ctx,ggml_mul_mat(ctx,weights.router,xn)));
            auto * xx=ggml_reshape_3d(ctx,xn,hidden,1,width);
            auto * ru=ggml_mul_mat_id(ctx,weights.up,xx,ids);
            auto * rg=ggml_mul_mat_id(ctx,weights.gate,xx,ids);
            auto * routed=ggml_reshape_2d(ctx,ggml_swiglu_split(ctx,rg,ru),ranks*middle,width);
            auto * su=ggml_mul_mat(ctx,weights.shared_up,xn);
            auto * sg=ggml_mul_mat(ctx,weights.shared_gate,xn);
            output=ggml_concat(ctx,routed,ggml_swiglu_split(ctx,sg,su),0);
        } else if (mode == Mode::router) {
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
            ggml_set_name(output, "n104_generic_q4");
        }
        ggml_build_forward_expand(graph, output);
        require(ggml_nelements(output) == (mode == Mode::phase ? (ranks+1)*middle*width : mode == Mode::router ? ranks * width : hidden * width), "graph output shape");
        int projections = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_MUL_MAT_ID) {
                ++projections;
                require(t->src[2]->ne[0] == ranks && t->src[2]->ne[1] == (mode == Mode::split ? 1 : width), "expert rank/width shape");
            }
        }
        require(projections == (mode == Mode::phase ? 2 : mode == Mode::router ? 0 : mode == Mode::generic ? 3 : 3*width), "graph projection count");
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

using LaunchCounter = unsigned long long (*)();
LaunchCounter cohort_count = nullptr;
LaunchCounter prior_q3_count = nullptr, xmad_count = nullptr;
using ReadIds = void (*)(int *);
ReadIds xmad_ids = nullptr;
using ReadPhase=void (*)(float *);
ReadPhase xmad_phase=nullptr;

void set_xmad(bool enabled) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_PHASEB_XMAD", enabled ? "1" : "0", 1)==0,"set xmad mode");
}

void set_cohorts(bool enabled) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_COHORTS", enabled ? "1" : "0", 1) == 0,
            "could not select q4 execution mode");
}
void load_counter() {
    void * handle = dlopen("/runtime/libggml-cuda.so", RTLD_NOLOAD | RTLD_NOW);
    require(handle, "candidate module not mapped");
    char executable[256]{};
    const auto n = readlink("/proc/self/exe", executable, sizeof(executable)-1);
    require(n > 0 && std::strcmp(executable, "/operator") == 0, "unexpected running operator binary");
    cohort_count = reinterpret_cast<LaunchCounter>(dlsym(handle, "ggml_cuda_occamy_q4_cohort_launch_count"));
    prior_q3_count = reinterpret_cast<LaunchCounter>(dlsym(handle,"ggml_cuda_occamy_q3_cohort_launch_count"));
    require(prior_q3_count && prior_q3_count()==0,"existing Q3 counter unavailable or dirty");
    require(cohort_count && cohort_count() == 0, "candidate launch counter unavailable or dirty");
    xmad_count=reinterpret_cast<LaunchCounter>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_count"));
    xmad_ids=reinterpret_cast<ReadIds>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_last_ids"));
    xmad_phase=reinterpret_cast<ReadPhase>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_last_phase"));
    require(xmad_count && xmad_ids && xmad_phase && xmad_count()==0,"missing actual physical counter or router reader");
    Dl_info xm_info{};
    require(dladdr(reinterpret_cast<void *>(xmad_count),&xm_info) && xm_info.dli_fname &&
        std::strcmp(xm_info.dli_fname,"/runtime/libggml-cuda.so")==0, "wrong xmad counter provider");
    Dl_info location{};
    require(dladdr(reinterpret_cast<void *>(cohort_count), &location) && location.dli_fname &&
            std::strcmp(location.dli_fname, "/runtime/libggml-cuda.so") == 0, "unexpected launch counter provider");
    std::fprintf(stderr, "Q4_COHORT MAP /operator counter_provider=%s\n", location.dli_fname);
    dlclose(handle);
}

struct Error {
    double rms, peak;
    int changed;
};
Error error(const Output & ref, const Output & actual, int row) {
    double diff2=0., magnitude=0., peak=0.;
    int changed=0;
    for (int i=row*hidden;i<(row+1)*hidden;++i) {
        require(std::isfinite(ref[i]) && std::isfinite(actual[i]),"nonfinite comparison row");
        const double d=double(actual[i])-ref[i];
        diff2+=d*d;
        magnitude+=double(ref[i])*ref[i];
        peak=std::max(peak,std::abs(d));
        changed+=std::memcmp(&ref[i],&actual[i],sizeof(float))!=0;
    }
    require(magnitude>0.,"empty comparison norm");
    return {std::sqrt(diff2/magnitude),peak,changed};
}
void compare(ggml_backend_t cuda, Graph & generic, Graph & split, Graph & router, Graph & phase, const Fixture & f) {
    set_cohorts(false);
    set_xmad(false);
    const auto initial_count = cohort_count();
    router.use(f);
    router.run(cuda);
    Ids ids{};
    for (int row = 0; row < width; ++row) {
        ggml_backend_tensor_get(router.output, ids.data() + row * ranks,
                                row * router.output->nb[1], ranks * sizeof(int32_t));
    }
    require(ids == f.ids, "device router IDs differ from seeded reference");
    std::fprintf(stderr, "N104 ROUTER %s ids=%zu/%zu\n", f.name, ids.size(), f.ids.size());
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
    require(cohort_count() == initial_count, "candidate active in a control graph");
    Output cohorts{};
    generic.use(f); // Restore row input in case the allocation plan allowed a residual alias.
    set_cohorts(true);
    generic.run(cuda);
    ggml_backend_tensor_get(generic.output, cohorts.data(), 0, sizeof(cohorts));
    require(cohort_count() == initial_count + 1, "missing physical q4 cohort launch");
    std::fprintf(stderr, "Q4_COHORT DIAG %s physical=%llu\n", f.name, cohort_count()-initial_count);
    Output xmad{}, repeated{};
    set_xmad(true);
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","1",1)==0,"activate direct route readback");
    auto initial_xmad=xmad_count();
    bool observed=true;
    for (Output * out : {&xmad,&repeated}) {
        generic.use(f);
        generic.run(cuda);
        ggml_backend_tensor_get(generic.output,out->data(),0,sizeof(Output));
        require(xmad_count()==++initial_xmad && cohort_count()==initial_count+1,"candidate or control physical coverage");
        Ids actual{};
        xmad_ids(actual.data());
        observed=observed && (actual==f.ids);
    }
    std::array<float,width*(ranks+1)*middle> observed_phase{},reference_phase{};
    xmad_phase(observed_phase.data());
    set_xmad(false);
    set_cohorts(false);
    phase.use(f);
    phase.run(cuda);
    ggml_backend_tensor_get(phase.output,reference_phase.data(),0,sizeof(reference_phase));
    for(int row=0;row<width;++row) {
        std::fprintf(stderr,"Q4_XMAD EXPERTS %s row=%d ids=",f.name,row);
        Ids selected{};
        xmad_ids(selected.data());
        for(int k=0;k<ranks;++k) std::fprintf(stderr,"%s%d",k?",":"",selected[row*ranks+k]);
        std::fprintf(stderr,"\n");
        double sq=0.,ref=0.,peak=0.;
        for(int i=row*(ranks+1)*middle;i<(row+1)*(ranks+1)*middle;++i) {
            require(std::isfinite(observed_phase[i]) && std::isfinite(reference_phase[i]),"nonfinite phase B");
            const double d=double(observed_phase[i])-reference_phase[i];
            sq+=d*d;
            ref+=double(reference_phase[i])*reference_phase[i];
            peak=std::max(peak,std::abs(d));
        }
        require(ref>0.,"degenerate phase B");
        std::fprintf(stderr,"Q4_XMAD PHASE %s row=%d finite=4608 generic_rms=%.12g generic_peak=%.12g\n",
                     f.name,row,std::sqrt(sq/ref),peak);
    }
    const bool identical=std::memcmp(xmad.data(),repeated.data(),sizeof(Output))==0;
    std::fprintf(stderr,"Q4_XMAD ROUTE %s matched=%d repeat_equal=%d launches=2\n",f.name,observed?32:0,int(identical));
    require(observed && identical,"physical xmad route or repeat result differs");
    set_xmad(false);
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","0",1)==0,"clear direct route readback");
    set_cohorts(false);
    for (int row = 0; row < width; ++row) {
        int different = 0;
        for (int i = row * hidden; i < (row + 1) * hidden; ++i) {
            require(std::isfinite(cohorts[i]) && std::isfinite(candidate[i]), "nonfinite q4 or sequential output");
            different += std::memcmp(&cohorts[i], &candidate[i], sizeof(float)) != 0;
        }
        std::fprintf(stderr, "Q4_COHORT ROW %s row=%d finite=2048 different=%d\n", f.name, row, different);
        require(different == 0, "q4 cohort differs from four original V3 row launches");
    }
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
        const Error old=error(base,cohorts,row), generic_err=error(base,xmad,row), v3_err=error(cohorts,xmad,row);
        std::fprintf(stderr,"Q4_XMAD ROW %s row=%d finite=2048 old_rms=%.12g old_peak=%.12g generic_rms=%.12g generic_peak=%.12g v3_rms=%.12g v3_peak=%.12g changed_v3=%d\n",
            f.name,row,old.rms,old.peak,generic_err.rms,generic_err.peak,v3_err.rms,v3_err.peak,v3_err.changed);
        require(magnitude > 0, "degenerate output");
        std::fprintf(stderr, "N104 NUMERIC %s row=%d finite=2048 mismatches=%d max_abs=%.9g relative_rms=%.9g\n",
                     f.name, row, mismatches, max_abs, std::sqrt(delta / magnitude));
    }
}

void perf(ggml_backend_t cuda, Graph & generic, Graph &, const Fixture & f) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","0",1)==0,"disable route readback");
    set_cohorts(true);
    auto batch=[&](bool xmad,int count) {
        set_xmad(xmad);
        generic.use(f);
        const auto old_count=cohort_count(), new_count=xmad_count();
        const auto start=std::chrono::steady_clock::now();
        for (int i=0;i<count;++i) generic.run(cuda);
        const auto finish=std::chrono::steady_clock::now();
        const auto old_calls=cohort_count()-old_count, new_calls=xmad_count()-new_count;
        Output input_after{};
        ggml_backend_tensor_get(generic.input,input_after.data(),0,sizeof(Output));
        require(std::memcmp(input_after.data(),f.input.data(),sizeof(Output))==0,"timing input changed");
        require(old_calls==unsigned(xmad?0:count) && new_calls==unsigned(xmad?count:0),"timing physical coverage");
        const double us=std::chrono::duration<double,std::micro>(finish-start).count()/count;
        std::fprintf(stderr,"Q4_XMAD TIME %s mode=%s reps=%d old=%llu new=%llu us=%.6f\n",
                     f.name,xmad?"xmad":"cohorts",count,old_calls,new_calls,us);
    };
    batch(false,8); batch(true,8);
    batch(false,reps); batch(true,reps); batch(true,reps); batch(false,reps);
    set_xmad(false);
    set_cohorts(false);
}
} // namespace

int main(int argc, char ** argv) {
    require(argc == 2, "expected --fixture, --check, or --perf");
    const bool offline = std::strcmp(argv[1], "--fixture") == 0;
    const bool diagnostic = std::strcmp(argv[1], "--check") == 0;
    require(offline || diagnostic || std::strcmp(argv[1], "--perf") == 0, "unexpected test mode");
    const WeightData data;
    const std::array<Fixture, 2> fixtures = {{{"independent", data, 1.0f}, {"partial", data, .34f}}};
    for (const auto & f : fixtures) { f.report(); }
    const auto & independent = fixtures[0].pair_overlap;
    const auto & partial = fixtures[1].pair_overlap;
    require(*std::max_element(independent.begin(), independent.end()) <= 2, "independent rows overlap too much");
    require(*std::min_element(partial.begin(), partial.end()) >= 2 &&
            *std::max_element(partial.begin(), partial.end()) <= 7, "partial rows outside realistic overlap");
    ggml_backend_t cuda = nullptr;
    {
        Weights weights;
        Graph generic(weights, Mode::generic), split(weights, Mode::split), router(weights, Mode::router), phase(weights, Mode::phase);
        if (offline) { return 0; }
        ggml_backend_reg_t reg = ggml_backend_load("/runtime/libggml-cuda.so");
        require(reg && ggml_backend_reg_dev_count(reg) == 1, "expected one CUDA device");
        cuda = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
        require(cuda && std::strcmp(ggml_backend_name(cuda), "CUDA0") == 0, "expected CUDA0 backend");
        check_mappings();
        load_counter();
        weights.load(cuda, data);
        generic.allocate(cuda);
        split.allocate(cuda);
        router.allocate(cuda);
        if (diagnostic) phase.allocate(cuda);
        for (const auto & f : fixtures) {
            if (diagnostic) { compare(cuda, generic, split, router, phase, f); }
            else { perf(cuda, generic, split, f); }
        }
    }
    require(prior_q3_count()==0, "unexpected Q3 candidate on Q4-only fixture");
    std::fprintf(stderr,"Q4_COHORT FINISH q3_launches=%llu q4_launches=%llu\n",prior_q3_count(),cohort_count());
    std::fprintf(stderr,"Q4_XMAD FINISH launches=%llu\n",xmad_count());
    ggml_backend_free(cuda);
}
