from pathlib import Path
import difflib
base=Path('occamy-fused.base.cu').read_text()
s=base.replace("#include <algorithm>","#include <algorithm>\n#include <atomic>")
a=s.index('template <bool L3>\n__global__ void __launch_bounds__(NT3, 2) moe_v3(');b=s.index('__global__ void moe_v3_cmp',a)
old=s[a:b];body=old.replace('template <bool L3>\n','').replace('moe_v3(Params P) {','moe_q2_cohorts(Params P, bool capture_ids) {\n    const int physical=blockIdx.y*gridDim.x+blockIdx.x;\n    const int row=physical%2, local_block=physical/2;\n    P.h+=row*H; P.out+=row*H; P.s+=row;')
body=body.replace('blockIdx.x * NW3','local_block * NW3').replace('warp * gridDim.x + blockIdx.x','warp * gridDim.x + local_block').replace('dot2c<L3','dot2c<true')
pat='if (warp == 0) route(s, lane, sel, selw);\n        __syncthreads();'
assert body.count(pat)==1
body=body.replace(pat,pat+'\n        if (capture_ids && local_block==0 && threadIdx.x<TOPK) s->logits[threadIdx.x]=float(sel[threadIdx.x]);')
s=s[:b]+body+s[b:]
a=s.index('static void reject(')
s=s[:a]+'''static std::atomic<unsigned long long> q2_calls{0}, q2_rows[2]{}, q2_skips{0};
static thread_local int observed_ids[2*TOPK];
static bool admit_q2(ggml_backend_cuda_context & ctx, const ggml_cuda_occamy_moe_match & m) {
    if (!env_flag("GGML_CUDA_OCCAMY_MOE_Q2_COHORTS") || ctx.curr_stream_no!=0 ||
            !ctx.stream_context().concurrent_events.empty() || ggml_cuda_info().devices[ctx.device].cc!=520 ||
            !ggml_cuda_occamy_n2split() || !m.h || !m.out) return false;
    const ggml_tensor * tensors[]={m.h,m.out};
    for(const ggml_tensor * t : tensors) {
        if(t->type!=GGML_TYPE_F32 || t->ne[0]!=H || t->ne[1]!=2 || t->ne[2]!=1 || t->ne[3]!=1 ||
           t->nb[1]!=H*sizeof(float) || !ggml_is_contiguous(t) || !t->data) return false;
    }
    const uintptr_t h=(uintptr_t)m.h->data, out=(uintptr_t)m.out->data;
    const uintptr_t delta=h>=out ? h-out : out-h;
    if(delta && delta<2*H*sizeof(float)) return false;
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool ready[GGML_CUDA_MAX_DEVICES]{};
    const int d=ctx.device;
    std::call_once(once[d],[d]{
        ggml_cuda_set_device(d);
        int sm=0, per=0;
        cudaFuncAttributes attr{};
        CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d));
        CUDA_CHECK(cudaFuncGetAttributes(&attr,moe_q2_cohorts));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,moe_q2_cohorts,NT3,SMEM3_FLOATS*sizeof(float)));
        ready[d]=sm==24 && per>=2 && sm*per>=48 && attr.numRegs<=64 && attr.localSizeBytes==0;
        fprintf(stderr,"Q2_COHORT residency device=%d sm=%d active=%d regs=%d static=%zu dynamic=%zu local=%zu ready=%d\\n",
            d,sm,per,attr.numRegs,attr.sharedSizeBytes,SMEM3_FLOATS*sizeof(float),attr.localSizeBytes,int(ready[d]));
    });
    return ready[d];
}
'''+s[a:]
a=s.index('bool ggml_cuda_occamy_enabled()')
s=s[:a]+'''extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q2_counter(unsigned which);
extern "C" unsigned long long ggml_cuda_occamy_q2_counter(unsigned which) {
    return which<2 ? occamy::q2_rows[which].load(std::memory_order_relaxed) :
           which==2 ? occamy::q2_calls.load(std::memory_order_relaxed) : occamy::q2_skips.load(std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default"))) void ggml_cuda_occamy_q2_router(int * out);
extern "C" void ggml_cuda_occamy_q2_router(int * out) { std::copy_n(occamy::observed_ids,2*occamy::TOPK,out); }
'''+s[a:]
a=s.index('    if (!v3 || d.grid_v3 <= 0)',s.index('void ggml_cuda_occamy_moe_launch('))
s=s[:a]+'''    // The native N2SPLIT caller makes adjacent row=0 and row=1 calls with the same match.
    // Suppress only its second callback after the first callback really launched both rows.
    struct Pending { const void * ctx=nullptr; const void * match=nullptr; const void * h=nullptr; const void * out=nullptr; };
    static thread_local Pending pending;
    if (row==1 && !out_override && m.nt==2 && pending.ctx==&ctx && pending.match==&m && pending.h==m.h && pending.out==m.out) {
        pending={}; q2_skips.fetch_add(1,std::memory_order_relaxed); return;
    }
    pending={};
    if (v3 && d.grid_v3>0 && row==0 && !out_override && m.nt==2 && admit_q2(ctx,m)) {
        ggml_cuda_pool_alloc<Scratch> work(ctx.pool(),2);
        static_assert(offsetof(Scratch,bar_gen)==offsetof(Scratch,bar_count)+sizeof(unsigned),"adjacent barrier words");
        CUDA_CHECK(cudaMemset2DAsync((char*)work.get()+offsetof(Scratch,bar_count),sizeof(Scratch),0,2*sizeof(unsigned),2,ctx.stream()));
        Params p{};
        p.L={(const float*)m.gate_inp->data,(const float*)m.gate_inp_sh->data,
             (const uint8_t*)m.gate_exps->data,(const uint8_t*)m.up_exps->data,(const uint8_t*)m.down_exps->data,
             (const uint8_t*)m.gate_sh->data,(const uint8_t*)m.up_sh->data,(const uint8_t*)m.down_sh->data};
        p.h=(const float*)m.h->data;p.norm=(const float*)m.norm_w->data;p.out=(float*)m.out->data;p.s=work.get();p.eps=m.eps;
        const bool diag=env_flag("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS");
        moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);
        CUDA_CHECK(cudaGetLastError());
        if (diag) {
            float values[2*TOPK];
            CUDA_CHECK(cudaMemcpy2DAsync(values,TOPK*sizeof(float),work.get(),sizeof(Scratch),
                TOPK*sizeof(float),2,cudaMemcpyDeviceToHost,ctx.stream()));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            for(int i=0;i<2*TOPK;++i) observed_ids[i]=int(values[i]);
        }
        q2_calls.fetch_add(1,std::memory_order_relaxed);
        pending={&ctx,&m,m.h,m.out};
        return;
    }
'''+s[a:]
pat='moe_v3<true><<<d.grid_v3, NT3, SMEM3_FLOATS * sizeof(float), st>>>(P);\n    CUDA_CHECK(cudaGetLastError());'
assert s.count(pat)==1
s=s.replace(pat,pat+'''\n    if (m.nt==2 && (row==0 || row==1) && env_flag("GGML_CUDA_OCCAMY_MOE_Q2_COUNT")) q2_rows[row].fetch_add(1,std::memory_order_relaxed);''')
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
