import difflib
from pathlib import Path
base=Path('occamy-fused.base.cu').read_text()
start=base.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end=base.index('__global__ void moe_v3_cmp', start)
kernel=base[start:end].replace('moe_q2_cohorts','moe_q2_direct_down')
a=kernel.index('    // C: down.')
barrier=kernel.index('    grid_barrier(s);',a)
prefetch='''    // Exact Phase C: direct 36-byte lane packets remove the shared transpose.
    int r = warp * gridDim.x + local_block;
    Q9 first{};
    if (r < H) first=load_down_q9(L,sel,r,lane,0);
    float gl = 0.f;
    {
        float gp[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) gp[q] = __ldcg(&s->gpart[q]);
#pragma unroll
        for (int q = 0; q < NQ; ++q) gl += gp[q];
    }
'''
kernel=kernel[:a]+prefetch+kernel[barrier:]
a=kernel.index('    for (; r < H; r += W)',a)
kernel=kernel[:a]+'''    for (; r < H; r += W) {
        const Q9 second=load_down_q9(L,sel,r,lane,1);
        const float dA=dot2c<true,true>(first,xs+lane*CH,sums[lane]);
        Q9 shared{};
        if (lane<8) shared=load_shared_q9(L,r,lane);
        const float dB=dot2c<true,true>(second,xs+(lane+32)*CH,sums[lane+32]);
        if (r+W<H) first=load_down_q9(L,sel,r+W,lane,0);
        float acc=__fmaf_rn(w0,dA,__fmul_rn(w1,dB));
        if (lane<8) acc=__fmaf_rn(gs,dot2c<true,true>(shared,xs+(lane+64)*CH,sums[lane+64]),acc);
        acc=wsum(acc);
        if (lane==0) P.out[r]=P.h[r]+acc;
        __syncwarp();
    }
}
'''
helper='''// Eight compute lanes consume 36 bytes each: two Q4_0 blocks per original dot2c.
// Select the same nine aligned words without an inter-lane shared transpose.
static_assert(ROW_D==288 && ROW_D%4==0);
__device__ __forceinline__ Q9 read_q9(const uint8_t * bytes) {
    const unsigned * p=reinterpret_cast<const unsigned *>(bytes);
    Q9 q;
#pragma unroll
    for (int i=0;i<9;++i) q.w[i]=__ldg(p+i);
    return q;
}
__device__ __forceinline__ Q9 load_down_q9(const Layer & L,const int * sel,int r,int lane,int c) {
    const int expert=sel[c*4+(lane>>3)];
    return read_q9(L.down_exps+((size_t)expert*H+r)*ROW_D+(lane&7)*36);
}
__device__ __forceinline__ Q9 load_shared_q9(const Layer & L,int r,int lane) {
    return read_q9(L.down_sh+(size_t)r*ROW_D+lane*36);
}
'''
s=base[:end]+helper+kernel+base[end:]
t=Path('../resume-q2-phaseb-pair/occamy-fused.cu').read_text()
a=t.index('static std::atomic<unsigned long long> gate_up_calls')
b=t.index('static bool admit_q2(',a)
admit=t[a:b].replace('gate_up','direct_down').replace('GATE_UP','DIRECT_DOWN').replace('GATEUP','DIRECTDOWN')
i=s.index('static bool admit_q2(');s=s[:i]+admit+s[i:]
launch='moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
assert s.count(launch)==1
s=s.replace(launch,'if (admit_direct_down(ctx.device)) {\n            '+launch.replace('moe_q2_cohorts','moe_q2_direct_down')+'\n            direct_down_calls.fetch_add(1,std::memory_order_relaxed);\n        } else '+launch)
Path('occamy-fused.cu').write_text(s)
Path('candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/occamy-fused.cu',tofile='b/occamy-fused.cu')))
