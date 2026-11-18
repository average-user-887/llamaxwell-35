from pathlib import Path
import difflib
s=Path('occamy-fused.base.cu').read_text()
s=s.replace('#include "occamy-fused.cuh"','#include "occamy-fused.cuh"\n#include "sm52-xmad.cuh"')
a=s.index('template <bool L3>\n__device__ __forceinline__ void moe_v3_body')
b=s.index('template <bool L3>\n__global__',a)
original=s[a:b]
new=original.replace('template <bool L3>\n','').replace('moe_v3_body','moe_q4_xmad_body').replace('dot2c<L3, true>','dot2c<true, true>')
old='const float * xc = xs + lane * CH; const float2 sx = sums[lane];'
assert new.count(old)==1
new=new.replace(old,'''if (local_block==0 && threadIdx.x<TOPK) s->logits[threadIdx.x]=float(sel[threadIdx.x]);
        auto * packed=reinterpret_cast<uint4 *>(xs+XM_PACK_FLOAT);
        auto * scale=reinterpret_cast<float2 *>(xs+XM_SCALE_FLOAT);
        xm_prepare(xs,packed,scale);''')
assert new.count('dot2c<L3>(lds9(stg + lane * 9), xc, sx)')==2
new=new.replace('dot2c<L3>(lds9(stg + lane * 9), xc, sx)','xm_dot(stg+lane*9, packed, scale, lane)')
s=s[:b]+Path('xmad.inc').read_text()+'\n'+new+s[b:]
a=s.index('__global__ void __launch_bounds__(NT3, 2) moe_v3_q4_cohorts'); b=s.index('__global__ void moe_v3_cmp',a)
s=s[:b]+s[a:b].replace('moe_v3_q4_cohorts','moe_q4_phaseb_xmad').replace('moe_v3_body<true>','moe_q4_xmad_body')+s[b:]
a=s.index('static bool q4_resident(');b=s.index('static void reject(',a)
new=s[a:b].replace('q4_resident','xm_resident').replace('moe_v3_q4_cohorts','moe_q4_phaseb_xmad').replace('Q4_COHORT residency','Q4_XMAD residency').replace('nsm*per_sm >= 48','nsm*per_sm >= 48 && attr.numRegs <= 64 && attr.localSizeBytes == 0')
s=s[:b]+new+'''static std::atomic<unsigned long long> xm_launches{0};
static int xm_report_ids[4*TOPK];
static float xm_report_phase[4*(TOPK*FF+FFS)];
'''+s[b:]
a=s.index('bool ggml_cuda_occamy_enabled()')
s=s[:a]+'''extern "C" __attribute__((visibility("default"))) void ggml_cuda_occamy_q4_xmad_last_phase(float * result);
extern "C" void ggml_cuda_occamy_q4_xmad_last_phase(float * result) { std::copy_n(occamy::xm_report_phase,4*(occamy::TOPK*occamy::FF+occamy::FFS),result); }
extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q4_xmad_count();
extern "C" unsigned long long ggml_cuda_occamy_q4_xmad_count() { return occamy::xm_launches.load(std::memory_order_relaxed); }
extern "C" __attribute__((visibility("default"))) void ggml_cuda_occamy_q4_xmad_last_ids(int * result);
extern "C" void ggml_cuda_occamy_q4_xmad_last_ids(int * result) { std::copy_n(occamy::xm_report_ids,32,result); }
'''+s[a:]
old='moe_v3_q4_cohorts<<<dim3(12,4), NT3, SMEM3_FLOATS*sizeof(float), ctx.stream()>>>(P);'
assert s.count(old)==1
s=s.replace(old,'''if (env_flag("GGML_CUDA_OCCAMY_MOE_Q4_PHASEB_XMAD") && xm_resident(ctx.device)) {
            moe_q4_phaseb_xmad<<<dim3(12,4),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(P);
            CUDA_CHECK(cudaGetLastError());
            xm_launches.fetch_add(1,std::memory_order_relaxed);
            if (env_flag("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG")) {
                float measured[4*TOPK];
                CUDA_CHECK(cudaMemcpy2DAsync(measured,TOPK*sizeof(float),workspace.get(),sizeof(Scratch),
                          TOPK*sizeof(float),4,cudaMemcpyDeviceToHost,ctx.stream()));
                constexpr size_t rowb=(TOPK*FF+FFS)*sizeof(float);
                CUDA_CHECK(cudaMemcpy2DAsync(xm_report_phase,rowb,(const char*)workspace.get()+offsetof(Scratch,h_exp),
                          sizeof(Scratch),TOPK*FF*sizeof(float),4,cudaMemcpyDeviceToHost,ctx.stream()));
                CUDA_CHECK(cudaMemcpy2DAsync(xm_report_phase+TOPK*FF,rowb,(const char*)workspace.get()+offsetof(Scratch,h_sh),
                          sizeof(Scratch),FFS*sizeof(float),4,cudaMemcpyDeviceToHost,ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                for (int i=0;i<4*TOPK;++i) xm_report_ids[i]=int(measured[i]);
            }
            return;
        }
        '''+old)
Path('occamy-fused.cu').write_text(s)
base=Path('occamy-fused.base.cu').read_text()
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
