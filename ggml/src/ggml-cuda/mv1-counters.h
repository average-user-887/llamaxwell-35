// MV2: dispatch/shape counters, default off. GGML_CUDA_MV1=1 counters only, =2 also per-FA-call cudaEvent timing (adds a sync per call).
// Output: GGML_CUDA_MV1_OUT=<file> (rewritten every 256 events, on normal exit, and by mv1::flush()); stderr when unset.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
namespace mv1 {
struct stat_t { long n = 0; double ms = 0; };
struct reg_t {
    std::map<std::string, stat_t> m; std::mutex mu; long events = 0; int mode = 0; const char * path = nullptr;
    reg_t() { const char * e = getenv("GGML_CUDA_MV1"); if (e && e[0] && e[0] != '0') { mode = atoi(e); if (mode == 0) mode = 1; } path = getenv("GGML_CUDA_MV1_OUT"); }
    void dump_locked() {
        FILE * f = (path && path[0]) ? fopen(path, "w") : stderr; if (!f) return;
        for (auto & kv : m) fprintf(f, "mv1d\t%s\tn=%ld\tms=%.3f\n", kv.first.c_str(), kv.second.n, kv.second.ms);
        fflush(f); if (f != stderr) fclose(f);
    }
    void add(const std::string & k, double ms = 0) {
        std::lock_guard<std::mutex> g(mu); auto & s = m[k]; s.n++; s.ms += ms; if (++events % 256 == 0) dump_locked();
    }
    void flush() { std::lock_guard<std::mutex> g(mu); dump_locked(); }
    ~reg_t() { if (mode) { dump_locked(); } }
};
inline reg_t & R() { static reg_t r; return r; }
inline int mode() { static const int v = R().mode; return v; }
inline std::string & tag() { static thread_local std::string t; return t; }   // per-graph tag "g<nodes>/t<ntok>"
}
