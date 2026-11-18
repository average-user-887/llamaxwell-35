from pathlib import Path
src=Path('../resume-q2-moe-cohorts/occamy-fused.cu').read_text()
start=src.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end=src.index('__global__ void moe_v3_',start)
k=src[start:end].replace('moe_q2_cohorts','moe_q2_shared_weights')
a=k.index('    // B:'); b=k.index('    // C:',a)
k=(k[:a]+Path('phase-b.inc').read_text()+k[b:]).replace('grid_barrier(s);','weights_barrier(P.s-row);')
bar=src[src.index('__device__ __forceinline__ void grid_barrier'):src.index('__device__ __forceinline__ float nib')]
bar=bar.replace('grid_barrier','weights_barrier').replace('gridDim.x - 1','gridDim.x * gridDim.y - 1').replace('    __syncthreads();','    __threadfence();\n    __syncthreads();',1)
src=src[:end]+bar+k+src[end:]
pos=src.index('static void reject(')
admit=src[src.index('static bool admit_q2('):pos]
start=admit.index('    static std::once_flag')
admit='static bool admit_shared(ggml_backend_cuda_context & ctx) {\n'+admit[start:]
admit=admit.replace('moe_q2_cohorts','moe_q2_shared_weights').replace('Q2_COHORT residency','Q2_WEIGHTS residency')
src=src[:pos]+'static std::atomic<unsigned long long> weight_calls{0};\n'+admit+src[pos:]
launch='        moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
src=src.replace(launch,'''        if (env_flag("GGML_CUDA_OCCAMY_MOE_Q2_SHARED_WEIGHTS") && admit_shared(ctx)) {
            moe_q2_shared_weights<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);
            weight_calls.fetch_add(1,std::memory_order_relaxed);
        } else {
'''+launch+'\n        }')
src=src.replace('return which<2 ?', 'if (which==4) return occamy::weight_calls.load(std::memory_order_relaxed);\n    return which<2 ?')
Path('occamy-fused.cu').write_text(src)
