import difflib
from pathlib import Path
base=Path('occamy-fused.base.cu').read_text()
start=base.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end=base.index('__global__ void moe_v3_cmp', start)
kernel=base[start:end].replace('moe_q2_cohorts','moe_q2_gate_up')
a=kernel.index('            const float gg = wsum(dot2c<true>')
b=kernel.index('            if (lane == 0)',a)
kernel=kernel[:a]+'''            const Q9 gate=lds9(stg+lane*9);
            __syncwarp();
            st72(stg4,lane,cv);
            __syncwarp();
            const Q9 up=lds9(stg+lane*9);
            const float2 gu=gate_up_pair(gate,up,xc,sx);
            const float gg=wsum(gu.x), uu=wsum(gu.y);
            const int tn=t+W; have=tn<NTB;
            if (have) { rowB(L,sel,tn,g,u); ld72(g,lane,cv); }
'''+kernel[b:]
s=base[:end]+Path('pair.inc').read_text()+kernel+base[end:]
i=s.index('static bool admit_q2(')
s=s[:i]+'''static std::atomic<unsigned long long> gate_up_calls{0};
extern "C" __attribute__((visibility("default"))) unsigned long long ggml_cuda_occamy_q2_gate_up_count();
extern "C" unsigned long long ggml_cuda_occamy_q2_gate_up_count() {return gate_up_calls.load(std::memory_order_relaxed);}
static bool admit_gate_up(int d) {
    if (!env_flag("GGML_CUDA_OCCAMY_MOE_Q2_GATE_UP")) return false;
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool ready[GGML_CUDA_MAX_DEVICES]{};
    std::call_once(once[d],[d]{
        ggml_cuda_set_device(d);
        int sm=0, per=0; cudaFuncAttributes attr{};
        CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d));
        CUDA_CHECK(cudaFuncGetAttributes(&attr,moe_q2_gate_up));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per,moe_q2_gate_up,NT3,SMEM3_FLOATS*sizeof(float)));
        ready[d]=sm==24 && per>=2 && attr.numRegs<=64 && attr.localSizeBytes==0;
        fprintf(stderr,"Q2_GATEUP residency device=%d sm=%d active=%d regs=%d static=%zu dynamic=%zu local=%zu ready=%d\\n",
            d,sm,per,attr.numRegs,attr.sharedSizeBytes,SMEM3_FLOATS*sizeof(float),attr.localSizeBytes,int(ready[d]));
    });
    return ready[d];
}
'''+s[i:]
launch='moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
s=s.replace(launch,'if (admit_gate_up(ctx.device)) {\n            '+launch.replace('moe_q2_cohorts','moe_q2_gate_up')+'\n            gate_up_calls.fetch_add(1,std::memory_order_relaxed);\n        } else '+launch)
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
