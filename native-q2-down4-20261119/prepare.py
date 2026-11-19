import difflib
from pathlib import Path

base = Path('occamy-fused.base.cu').read_text()
start = base.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end = base.index('__global__ void moe_v3_cmp', start)
kernel = base[start:end].replace('moe_q2_cohorts','moe_q2_down4')
kernel = kernel[:kernel.index('    const float gs =')] + Path('down4-loop.inc').read_text()
kernel = kernel.replace('int r = warp * gridDim.x + local_block;', 'int r = 4 * (warp * gridDim.x + local_block);')
kernel = kernel.replace('if (r < H) ldD(L, sel, r, 0, lane, cv);', 'if (r < H) ld_down4(L, sel, r, 0, lane, cv);')
s = base[:end] + Path('down4.inc').read_text() + kernel + base[end:]
i = s.index('static bool admit_q2(')
s = s[:i] + '''static std::atomic<unsigned long long> down4_calls{0};
extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q2_down4_count();
extern "C" unsigned long long ggml_cuda_occamy_q2_down4_count() {return down4_calls.load(std::memory_order_relaxed);}
static bool admit_down4(int d) {
    if (!env_flag("GGML_CUDA_OCCAMY_MOE_Q2_DOWN4")) return false;
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool ready[GGML_CUDA_MAX_DEVICES]{};
    std::call_once(once[d],[d]{
        ggml_cuda_set_device(d);
        int sm=0, per=0; cudaFuncAttributes attr{};
        CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d));
        CUDA_CHECK(cudaFuncGetAttributes(&attr,moe_q2_down4));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,moe_q2_down4,NT3,SMEM3_FLOATS*sizeof(float)));
        ready[d]=sm==24 && per>=2 && attr.numRegs<=64 && attr.localSizeBytes==0;
        fprintf(stderr,"Q2_DOWN4 residency device=%d sm=%d active=%d regs=%d static=%zu dynamic=%zu local=%zu ready=%d\\n",
            d,sm,per,attr.numRegs,attr.sharedSizeBytes,SMEM3_FLOATS*sizeof(float),attr.localSizeBytes,int(ready[d]));
    });
    return ready[d];
}
''' + s[i:]
launch = 'moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
assert s.count(launch) == 1
s = s.replace(launch, 'if (admit_down4(ctx.device)) {\n            '+launch.replace('moe_q2_cohorts','moe_q2_down4')+'\n            down4_calls.fetch_add(1,std::memory_order_relaxed);\n        } else '+launch)
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
