import difflib
from pathlib import Path
base=Path('occamy-fused.base.cu').read_text()
start=base.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end=base.index('__global__ void moe_v3_cmp', start)
kernel=base[start:end].replace('moe_q2_cohorts','moe_q2_offsets')
kernel=kernel.replace('__shared__ int sel[TOPK];', '__shared__ int sel[TOPK];\n    __shared__ uint32_t expert_offsets[TOPK];')
kernel=kernel.replace('if (warp == 0) route(s, lane, sel, selw);', """if (warp == 0) {
            route(s, lane, sel, selw);
            if (lane < TOPK) expert_offsets[lane] = uint32_t(sel[lane]) * EXPERT_PITCH;
        }""")
for task in ('gw','t','tn'):
    kernel=kernel.replace('rowB(L, sel, '+task+', g, u)', 'rowB_offset(L, expert_offsets, '+task+', lane, g, u)')
kernel=kernel.replace('ld72(', 'load_gate_up_offset(').replace('ldD(L, sel,', 'load_down_offset(L, expert_offsets,')
s=base[:end]+Path('offsets.inc').read_text()+kernel+base[end:]
i=s.index('static bool admit_q2(')
s=s[:i]+'''static std::atomic<unsigned long long> offsets_calls{0};
extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q2_offsets_count();
extern "C" unsigned long long ggml_cuda_occamy_q2_offsets_count() {return offsets_calls.load(std::memory_order_relaxed);}
static bool admit_offsets(int d) {
    if (!env_flag("GGML_CUDA_OCCAMY_MOE_Q2_OFFSETS")) return false;
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool ready[GGML_CUDA_MAX_DEVICES]{};
    std::call_once(once[d],[d]{
        ggml_cuda_set_device(d);
        int sm=0, per=0; cudaFuncAttributes attr{};
        CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d));
        CUDA_CHECK(cudaFuncGetAttributes(&attr,moe_q2_offsets));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,moe_q2_offsets,NT3,SMEM3_FLOATS*sizeof(float)));
        ready[d]=sm==24 && per>=2 && attr.numRegs<=64 && attr.localSizeBytes==0;
        fprintf(stderr,"Q2_OFFSETS residency device=%d sm=%d active=%d regs=%d static=%zu dynamic=%zu local=%zu ready=%d\\n",
            d,sm,per,attr.numRegs,attr.sharedSizeBytes,SMEM3_FLOATS*sizeof(float),attr.localSizeBytes,int(ready[d]));
    });
    return ready[d];
}
'''+s[i:]
launch='moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
s=s.replace(launch,'if (admit_offsets(ctx.device)) {\n            '+launch.replace('moe_q2_cohorts','moe_q2_offsets')+'\n            offsets_calls.fetch_add(1,std::memory_order_relaxed);\n        } else '+launch)
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
