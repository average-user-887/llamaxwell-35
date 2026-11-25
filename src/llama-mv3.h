#pragma once
// MV3: LLAMA_MV3_PHASE=<file> (default off). Per-call host phase timing around llama decode, keyed by (ctx role, n_tokens).
// Zero cost when off: every hook is guarded by mv3::on() (a cached bool).
#include "ggml.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace mv3 {

struct Rec { // per-ubatch phase accumulator (us)
    double prep = 0, apply = 0, graph = 0, setin = 0, compute = 0, fetch = 0, rs = 0;
    int reused = 0;
};

struct Row {
    long calls = 0, reused = 0, synced = 0;
    double prep = 0, apply = 0, graph = 0, setin = 0, compute = 0, launch = 0, sync = 0, fetch = 0, rs = 0, gpu_ms = 0;
    int n_layer = 0, n_embd = 0, type = 0;
};

typedef std::pair<int, int> Key; // (ctx order, n_tokens)

struct State {
    const char * path = getenv("LLAMA_MV3_PHASE");
    bool on = path != nullptr && path[0] != 0;
    std::mutex mu;
    std::map<Key, Row> rows;
    long n = 0;
    int n_ctx = 0;
    void dump() { // caller holds mu
        FILE * f = fopen(path, "w"); if (!f) return;
        fprintf(f, "#role\tn_tokens\tn_layer\tn_embd\tctx_type\tcalls\treused\tsynced"
                   "\tprep_us\tapply_us\tgraph_us\tsetin_us\tcompute_us\tlaunch_us\tsync_us\tfetch_us\trs_us\tgpu_span_ms"
                   "\tmean_prep_us\tmean_apply_us\tmean_graph_us\tmean_setin_us\tmean_compute_us\tmean_launch_us\tmean_sync_us\tmean_fetch_us\tmean_rs_us\tmean_gpu_span_ms\n");
        for (auto & kv : rows) {
            const Row & r = kv.second; const double c = r.calls ? (double) r.calls : 1, s = r.synced ? (double) r.synced : 1;
            fprintf(f, "ctx%d\t%d\t%d\t%d\t%d\t%ld\t%ld\t%ld\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.4f"
                       "\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\n",
                kv.first.first, kv.first.second, r.n_layer, r.n_embd, r.type, r.calls, r.reused, r.synced,
                r.prep, r.apply, r.graph, r.setin, r.compute, r.launch, r.sync, r.fetch, r.rs, r.gpu_ms,
                r.prep/c, r.apply/c, r.graph/c, r.setin/c, r.compute/c, r.launch/c, r.sync/s, r.fetch/c, r.rs/c, r.gpu_ms/s);
        }
        fclose(f);
    }
    ~State() { if (on) { std::lock_guard<std::mutex> l(mu); if (n) dump(); } }
};

inline State & S() { static State s; return s; }
inline bool on() { static const bool v = S().on; return v; }

// recurrent-state work outside the graph (seq_rm/seq_cp/find_slot), consumed by the next committed ubatch
inline double & rs_acc() { static thread_local double v = 0; return v; }
struct RsScope {
    int64_t t0; bool a;
    RsScope() : t0(0), a(on()) { if (a) t0 = ggml_time_us(); }
    ~RsScope() { if (a) rs_acc() += (double) (ggml_time_us() - t0); }
};

// per-context bookkeeping
struct Ctx {
    int order = -1;
    int n_layer = 0, n_embd = 0, type = 0;
    std::deque<Key> pending; // ubatches launched, not yet synchronized
    double prep_decode = 0;  // decode-level prep, charged to the first ubatch
    bool first = true;
    void init(int nl, int ne, int ty) {
        if (!on() || order >= 0) return;
        std::lock_guard<std::mutex> l(S().mu); order = S().n_ctx++; n_layer = nl; n_embd = ne; type = ty;
    }
    void commit(int n_tokens, const Rec & r) {
        std::lock_guard<std::mutex> l(S().mu);
        Key k(order, n_tokens); Row & w = S().rows[k];
        w.n_layer = n_layer; w.n_embd = n_embd; w.type = type;
        w.calls++; w.reused += r.reused;
        w.prep += r.prep; w.apply += r.apply; w.graph += r.graph; w.setin += r.setin; w.compute += r.compute;
        w.fetch += r.fetch; w.rs += r.rs;
        pending.push_back(k);
        if (++S().n % 50 == 0) S().dump();
    }
    // called after the existing sync returned: waited_us over all pending ubatches, spans[i] = GPU ms for pending i (or <0)
    void synced(double waited_us, const float * spans, const double * launch, int n_spans) {
        if (pending.empty()) return;
        std::lock_guard<std::mutex> l(S().mu);
        const double share = waited_us / (double) pending.size();
        for (size_t i = 0; i < pending.size(); i++) {
            Row & w = S().rows[pending[i]];
            w.sync += share; w.synced++;
            if ((int) i < n_spans) { if (spans[i] >= 0) w.gpu_ms += spans[i]; w.launch += launch[i]; }
        }
        pending.clear();
    }
};

// CUDA backend hooks (resolved through the backend reg proc address)
typedef void (*group_begin_t)(ggml_backend_t);
typedef int  (*group_read_t)(ggml_backend_t, float * span_ms, double * launch_us, int max);

} // namespace mv3
