// CPU fixture for the observer ABI and interposition only.
#include "ggml.h"
#include "../match.inc"
#include <cstdio>
struct ggml_backend_cuda_context {};
static unsigned long long calls;
extern "C" unsigned long long ggml_cuda_occamy_q4_cohort_launch_count() { return calls; }
void ggml_cuda_occamy_moe_launch(ggml_backend_cuda_context &, const ggml_cuda_occamy_moe_match & m, float *, int) {
    calls += m.nt==4;
}
bool ggml_cuda_occamy_moe_match_window(ggml_backend_cuda_context &,const ggml_cgraph *,int,ggml_cuda_occamy_moe_match &) { return true; }
extern "C" void run_fixture() {
    ggml_backend_cuda_context ctx;
    ggml_cuda_occamy_moe_match m;
    ggml_tensor w{};
    m.gate_inp=&w;
    for (int width=1;width<=4;++width) for (int i=0;i<40;++i) {
        std::snprintf(w.name,sizeof(w.name),"blk.%d.ffn_gate_inp.weight",i);
        m.nt=width; if (width==3) continue;
        if (ggml_cuda_occamy_moe_match_window(ctx,nullptr,0,m)) ggml_cuda_occamy_moe_launch(ctx,m,nullptr,0);
    }
    std::snprintf(w.name,sizeof(w.name),"blk.40.ffn_gate_inp.weight");
    if (ggml_cuda_occamy_moe_match_window(ctx,nullptr,0,m)) ggml_cuda_occamy_moe_launch(ctx,m,nullptr,0);
}
