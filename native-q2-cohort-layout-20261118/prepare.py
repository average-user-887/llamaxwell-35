"""Add one mapping-only kernel and two diagnostic witnesses to the existing two-row object."""
import difflib
from pathlib import Path
base=Path('occamy-fused.base.cu').read_text()
a=base.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
b=base.index('__global__ void moe_v3_cmp',a)
body=base[a:b]
original='const int physical=blockIdx.y*gridDim.x+blockIdx.x;\n    const int row=physical%2, local_block=physical/2;'
major='const int row=int(blockIdx.y), local_block=int(blockIdx.x);'
assert body.count(original)==1
marker='    P.h+=row*H; P.out+=row*H; P.s+=row;'
probe="""
    if (threadIdx.x == 0) {
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        float * record = P.s->logits + TOPK + 5 * local_block;
        record[0] = float(blockIdx.x);
        record[1] = float(blockIdx.y);
        record[2] = float(row);
        record[3] = float(local_block);
        record[4] = float(sm);
    }
"""
extra=[]
for name,row_major,tap in [('moe_q2_major',True,False),('moe_q2_map_old',False,True),('moe_q2_map_major',True,True)]:
    k=body.replace('moe_q2_cohorts',name)
    if row_major:k=k.replace(original,major)
    if tap:k=k.replace(marker,marker+probe)
    extra.append(k)
s=base[:b]+''.join(extra)+base[b:]
i=s.index('static bool admit_q2(')
s=s[:i]+'''static std::atomic<unsigned long long> layout_calls{0};
static thread_local float last_layout_map[2*24*5];
extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q2_layout_count();
extern "C" unsigned long long ggml_cuda_occamy_q2_layout_count() { return layout_calls.load(std::memory_order_relaxed); }
extern "C" __attribute__((visibility("default"))) void ggml_cuda_occamy_q2_layout_map(float * out);
extern "C" void ggml_cuda_occamy_q2_layout_map(float * out) { std::copy_n(last_layout_map,2*24*5,out); }

static bool admit_layout(int d,int kind) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES][3];
    static bool ready[GGML_CUDA_MAX_DEVICES][3]{};
    std::call_once(once[d][kind],[d,kind] {
        ggml_cuda_set_device(d);
        const void * kernel = kind==0 ? (const void *)moe_q2_major :
            (kind==1 ? (const void *)moe_q2_map_old : (const void *)moe_q2_map_major);
        int sm=0,per=0; cudaFuncAttributes attr{};
        CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d));
        CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,kernel,NT3,SMEM3_FLOATS*sizeof(float)));
        ready[d][kind]=sm==24 && per>=2 && attr.numRegs<=64 && attr.localSizeBytes==0;
        fprintf(stderr,"Q2_LAYOUT residency kind=%d device=%d sm=%d active=%d regs=%d static=%zu dynamic=%zu local=%zu ready=%d\\n",
                kind,d,sm,per,attr.numRegs,attr.sharedSizeBytes,SMEM3_FLOATS*sizeof(float),attr.localSizeBytes,int(ready[d][kind]));
    });
    return ready[d][kind];
}
'''+s[i:]
launch='moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
call=launch[launch.index('<<<'):]
dispatch='''const bool major=env_flag("GGML_CUDA_OCCAMY_MOE_Q2_LAYOUT");
        const bool map=diag && env_flag("GGML_CUDA_OCCAMY_MOE_Q2_LAYOUT_MAP");
        const int kind=map ? (major ? 2 : 1) : 0;
        const bool selected=(major || map) && admit_layout(ctx.device,kind);
        if (selected && map) {
            static_assert(TOPK+24*5<=NE);
            CUDA_CHECK(cudaMemset2DAsync((char *)work.get()+offsetof(Scratch,logits)+TOPK*sizeof(float),
                sizeof(Scratch),0xff,24*5*sizeof(float),2,ctx.stream()));
            if (major) moe_q2_map_majorCALL
            else moe_q2_map_oldCALL
        } else if (selected) moe_q2_majorCALL
        else moe_q2_cohortsCALL
        if (selected && major) layout_calls.fetch_add(1,std::memory_order_relaxed);
        if (selected && map) {
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy2DAsync(last_layout_map,24*5*sizeof(float),(char *)work.get()+offsetof(Scratch,logits)+TOPK*sizeof(float),
                sizeof(Scratch),24*5*sizeof(float),2,cudaMemcpyDeviceToHost,ctx.stream()));
        }'''.replace('CALL',call)
s=s.replace(launch,dispatch)
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
