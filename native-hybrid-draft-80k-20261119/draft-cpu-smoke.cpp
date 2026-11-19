#include "ggml.h"
#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// Compiled from the production row guard by cpu-check.sh.
#include "row-guard.inc"

static void check_width(int n) {
    constexpr int d = 256, nh = 16, nk = 2, nkv = 256;
    auto * ctx0 = ggml_init({ 24u << 20, nullptr, false });
    auto * gf = ggml_new_graph(ctx0);
    auto * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n);
    auto * pos = static_cast<int *>(inp_pos->data);
    for (int s=0; s<4; ++s) {
        for (int r=0; r<n; ++r) {
            pos[s*n+r] = s==3 ? 0 : 32760 + r;
        }
    }
    llama_ubatch ub{};
    llama_token tok[3] = {5, 6, 7};
    llama_seq_id seq = 0;
    llama_seq_id * seq_ids[3] = {&seq, &seq, &seq};
    int32_t counts[3] = {1, 1, 1};
    int8_t out[3] = {};
    out[n-1] = 1;
    float state = 0;
    ub.n_tokens=n; ub.n_pos=4; ub.n_seqs_unq=1; ub.pos=pos; ub.output=out; ub.seq_id=seq_ids;
    ub.token=tok; ub.embd_state=&state; ub.n_seq_id=counts;
    assert(mtp_last_query_batch(ub,1));
    out[n-1]=0; out[0]=1; assert(!mtp_last_query_batch(ub,1)); out[0]=0; out[n-1]=1;
    ++pos[n]; assert(!mtp_last_query_batch(ub,1)); --pos[n];
    ++pos[1]; assert(!mtp_last_query_batch(ub,1)); --pos[1];
    ++counts[0]; assert(!mtp_last_query_batch(ub,1)); --counts[0];
    ++ub.n_seqs_unq; assert(!mtp_last_query_batch(ub,1)); --ub.n_seqs_unq;
    assert(!mtp_last_query_batch(ub,0));
    // Use different values in the four position sections to expose a rolled or truncated gather.
    for (int s=0; s<4; ++s) {
        for (int r=0; r<n; ++r) { pos[s*n+r] += 17*s; }
    }
    auto * Q = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d, nh, n);
    for (int i=0; i<ggml_nelements(Q); ++i) { ((float *) Q->data)[i]=std::sin(0.013*i); }
    auto * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, nkv, n);
    for (int r=0; r<n; ++r) {
        for (int j=0; j<nkv; ++j) {
            ((ggml_fp16_t *)mask->data)[r*nkv+j]=ggml_fp32_to_fp16(j<=r+4 && j!=2 ? 0 : -INFINITY);
        }
    }
    auto * k = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, d, nkv, nk);
    auto * v = ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, d, nkv, nk);
    for (int i=0; i<ggml_nelements(k); ++i) {
        ((ggml_fp16_t *)k->data)[i]=ggml_fp32_to_fp16(std::sin(0.002*i));
        ((ggml_fp16_t *)v->data)[i]=ggml_fp32_to_fp16(std::cos(0.003*i));
    }
    auto attend = [&](ggml_tensor * q, ggml_tensor * p, ggml_tensor * m) {
        int sections[] = {11,11,10,0};
        q = ggml_rope_multi(ctx0,q,p,nullptr,64,sections,GGML_ROPE_TYPE_IMROPE,262144,1e7,1,0,1,32,1);
        q = ggml_permute(ctx0,q,0,2,1,3);
        return ggml_flash_attn_ext(ctx0,q,k,v,m,1.0f/std::sqrt(float(d)),0,0);
    };
    const int last=n-1;
    ggml_tensor * q_pos;
    const int n_tokens=n;
    // Compiled from the production section gather and mask view, apart from assignment of the latter.
    #include "graph-views.inc"
    auto * qall = attend(Q,inp_pos,mask);
    auto * qlast = ggml_view_3d(ctx0,Q,d,nh,1,Q->nb[1],Q->nb[2],last*Q->nb[2]);
    auto * one = attend(qlast,q_pos,last_mask);
    auto * bait = attend(qlast,q_pos,ggml_view_1d(ctx0,mask,nkv,0));
    ggml_build_forward_expand(gf,qall);
    ggml_build_forward_expand(gf,one);
    ggml_build_forward_expand(gf,bait);
    assert(ggml_graph_compute_with_ctx(ctx0,gf,2)==GGML_STATUS_SUCCESS);
    for(int j=0;j<4;++j) { assert(((int *)q_pos->data)[j]==pos[j*n+last]); }
    double diff=0, wrong=0;
    for(int i=0;i<d*nh;++i) {
        double ref=((float *)qall->data)[last*d*nh+i];
        diff=std::max(diff,std::abs(ref-((float *)one->data)[i]));
        wrong=std::max(wrong,std::abs(ref-((float *)bait->data)[i]));
    }
    assert(diff<0.001 && wrong>0.01);
    std::printf("n=%d four position sections exact, FA last-row max_abs=%g wrong-mask control=%g\n",n,diff,wrong);
    ggml_free(ctx0);
}
static int consumed_rows(int depth) {
    int n_drafting = 1, seq_id = 0;
    bool drafting[] = { true };
    struct { int n_max; } params{depth};
    std::vector<int> result;
    int consumed = 0;
    while (n_drafting) {
        result.push_back(13);
        #include "depth-check.inc"
        ++consumed;
    }
    assert(!drafting[0]);
    return consumed;
}
int main() {
    ggml_cpu_init(); check_width(2); check_width(3);
    assert(consumed_rows(1) == 0 && consumed_rows(2) == 1 && consumed_rows(3) == 2);
    std::puts("Native terminal branch: depth 1 reads zero next-head values, depth 2 consumes one, depth 3 consumes two.");
}
