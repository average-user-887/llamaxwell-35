from pathlib import Path
s=Path('../resume-q2-moe-cohorts/occamy-fused.cu').read_text()
start=s.index('__global__ void __launch_bounds__(NT3, 2) moe_q2_cohorts')
end=s.index('__global__ void moe_v3_',start)
k=s[start:end].replace('moe_q2_cohorts','moe_q2_router')
k=k[:k.index('    // A:')]+Path('phase-a.inc').read_text()+k[k.index('    // B:'):]
bar=s[s.index('__device__ __forceinline__ void grid_barrier'):s.index('__device__ __forceinline__ float nib')].replace('grid_barrier','router_barrier').replace('gridDim.x - 1','gridDim.x * gridDim.y - 1').replace('    __syncthreads();','    __threadfence();\n    __syncthreads();',1)
s=s[:end]+bar+k+s[end:]
ref=Path('../fresh-q2-phaseb-weights/occamy-fused.cu').read_text()
admit=ref[ref.index('static std::atomic<unsigned long long> weight_calls'):ref.index('static void reject(')].replace('moe_q2_shared_weights','moe_q2_router').replace('weight_calls','router_calls').replace('admit_shared','admit_router').replace('Q2_WEIGHTS','Q2_ROUTER')
pos=s.index('static void reject(');s=s[:pos]+admit+s[pos:]
s=s.replace('ggml_cuda_pool_alloc<Scratch> work(ctx.pool(),2);','const bool router=env_flag("GGML_CUDA_OCCAMY_MOE_Q2_ROUTER") && admit_router(ctx);\n        ggml_cuda_pool_alloc<Scratch> work(ctx.pool(),router ? 3 : 2);').replace('2*sizeof(unsigned),2,ctx.stream()', '2*sizeof(unsigned),router ? 3 : 2,ctx.stream()')
line='        moe_q2_cohorts<<<dim3(24,2),NT3,SMEM3_FLOATS*sizeof(float),ctx.stream()>>>(p,diag);'
assert s.count(line)==1
s=s.replace(line,'        if (router) {\n'+line.replace('moe_q2_cohorts','moe_q2_router')+'\n            router_calls.fetch_add(1,std::memory_order_relaxed);\n        } else {\n'+line+'\n        }').replace('return which<2 ?', 'if (which==4) return occamy::router_calls.load(std::memory_order_relaxed);\n    return which<2 ?')
Path('occamy-fused.cu').write_text(s)
