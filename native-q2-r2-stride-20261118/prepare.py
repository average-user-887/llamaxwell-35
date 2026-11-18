"""Build the narrowly dispatched retained-attention derivative from the measured probe baseline."""
import difflib
from pathlib import Path

here = Path(__file__).resolve().parent
base = here.parent / "last-ms-r2-probe"
name = "fattn-occamy-sparse.cuh"
old = (base / name).read_text()

def once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)

start = old.index("template <int NQ, int NU>\n__global__")
end = old.index("static std::atomic<long long> ofa_sparse_calls", start)
original_kernel = old[start:end]
k = once(original_kernel, """template <int NQ, int NU>
__global__ void __launch_bounds__(32*NU*(2*NQ), (NQ == 1 ? 2 : 1)) ofa_sparse_attention(const ofa_params p, ofa_sparse_plan * plans) {""", """template <unsigned TOKEN_BYTES>
__global__ void __launch_bounds__(512,1) ofa_sparse_attention_q2_stride(const ofa_params p, ofa_sparse_plan * plans) {
    constexpr int NQ = 2, NU = 4;
    static_assert(TOKEN_BYTES == 1024 || TOKEN_BYTES == 2048, "Unexpected retained KV layout");""")
k = once(k, """        if (use_sparse) atomicAdd(&plans[plan_index].sparse_warps, 1);
        else { atomicExch(&plans[plan_index].fallback, 1); atomicAdd(&plans[plan_index].full_warps, 1); }""", """        // Eight warps in this CTA validate exactly the same immutable row/head plan.
        if (head_group == 0 && ul == 0) {
            if (use_sparse) atomicAdd(&plans[plan_index].sparse_warps, 2*NU);
            else {
                atomicExch(&plans[plan_index].fallback, 1);
                atomicAdd(&plans[plan_index].full_warps, 2*NU);
            }
        }""")
k = once(k, """    const int u0 = (int) (((long long) split      * nunits) / p.nsplit);
    const int u1 = (int) (((long long)(split + 1) * nunits) / p.nsplit);""", """    // Host dispatch requires twelve splits and a maximum width of 83,200.
    const int u0 = split*nunits/12;
    const int u1 = (split + 1)*nunits/12;""")
# The source assigns blockIdx.z to a general sequence offset. The guarded
# grid below has z=1. A packed 256-element F16 head occupies 512 bytes.
k = once(k, "    const int i3    = blockIdx.z;", "    constexpr int i3 = 0;")
k = once(k, "const char * Kb = p.K + i3*p.nbk3 + kvh*p.nbk2 + lane*16;",
         "const char * Kb = p.K + unsigned(kvh)*512u + unsigned(lane)*16u;")
k = once(k, "const char * Vb = p.V + i3*p.nbv3 + kvh*p.nbv2 + lane*16;",
         "const char * Vb = p.V + unsigned(kvh)*512u + unsigned(lane)*16u;")
k = once(k, "const half * Mb = (const half *) (p.mask + (i3 % p.mask_ne3)*p.nbm3 + (size_t) rl*p.nbm1);",
         "const half * Mb = (const half *) (p.mask + (rl == 0 ? size_t(0) : p.nbm1));")
k = once(k, "    const int nunits = use_sparse ? plan.units : p.n_kv/OFA_T;", """    // Each lane holds one original far-block ID. The old IDs, boundary and
    // split counts are immutable for every warp consuming this plan.
    const int old_blocks = use_sparse ? plan.n_blocks : 0;
    const int old_units = old_blocks*(OFS_BLOCK/OFA_T);
    const int recent_begin = use_sparse ? plan.recent_begin : 0;
    const int lane_block = lane < old_blocks ? plan.blocks[lane] : 0;
    const int nunits = use_sparse ? plan.units : p.n_kv/OFA_T;""")
k = once(k, "        const int t0 = use_sparse ? ofa_sparse_token(plan, u) : u*OFA_T;", """        int t0 = u*OFA_T;
        if (use_sparse) {
            if (u < old_units) {
                const int b = __shfl_sync(0xffffffff, lane_block, u/(OFS_BLOCK/OFA_T), 32);
                t0 = b*OFS_BLOCK + (u%(OFS_BLOCK/OFA_T))*OFA_T;
            } else {
                t0 = recent_begin + (u-old_units)*OFA_T;
            }
        }""")
k = once(k, "    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;\n", "")
k = once(k, """        const char * kp = Kb + (size_t) t0*nbk1;
        const char * vp = Vb + (size_t) t0*nbv1;""", """        // Maximum unsigned token offset is below 171 MB. GPU base addresses
        // and the view-specific head and lane adjustments retain full width.
        const unsigned offset = unsigned(t0)*TOKEN_BYTES;
        const char * kp = Kb + offset;
        const char * vp = Vb + offset;""")
k = once(k, """            const char * r = kp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbk1) kr[j] = __ldg((const uint4 *) r);""", """#pragma unroll
            for (int j = 0; j < OFA_T; ++j) kr[j] = __ldg((const uint4 *) (kp + unsigned(j)*TOKEN_BYTES));""")
k = once(k, """            const char * r = vp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbv1) vr[j] = __ldg((const uint4 *) r);""", """#pragma unroll
            for (int j = 0; j < OFA_T; ++j) vr[j] = __ldg((const uint4 *) (vp + unsigned(j)*TOKEN_BYTES));""")
s = old[:end] + k + old[end:]
dispatch = """    ofa_sparse_attention<NQ, NU><<<dim3(nsplit, nkvh, 1), NT, 0, stream>>>(p, plans.get());"""
s = once(s,dispatch,"""    static const bool want_stride = [] {
        const char * flag = getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE");
        return flag && strcmp(flag, "1") == 0;
    }();
    const bool stride = tile4 && want_stride && nsplit == 12 && blocks_per_sm == 1 &&
            p.nbk2 == 512 && p.nbv2 == 512 && p.nbk1 == p.nbv1 && (p.nbk1 == 1024 || p.nbk1 == 2048);
    if (stride) {
        static std::atomic<int> stride_seen{0};
        if (stride_seen.fetch_add(1, std::memory_order_relaxed) == 0) {
            fprintf(stderr, "q2_r2_stride_dispatch width=%d k_token_bytes=%zu v_token_bytes=%zu splits=%d nu=4 threads=512 grid=%d\\n",
                    n_kv, p.nbk1, p.nbv1, nsplit, nsplit*nkvh);
        }
        if (p.nbk1 == 1024) ofa_sparse_attention_q2_stride<1024><<<dim3(nsplit,nkvh,1),NT,0,stream>>>(p,plans.get());
        else ofa_sparse_attention_q2_stride<2048><<<dim3(nsplit,nkvh,1),NT,0,stream>>>(p,plans.get());
    } else {
    """+dispatch+"""
    }""")
check = "            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && output_diff == 0 && nonfinite == 0);"
s = once(s, check, check+"\n"+(here/"dense-check.inc").read_text())
(here/name).write_text(s)
p = "ggml/src/ggml-cuda/"+name
(here/"candidate.patch").write_text("".join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),"a/"+p,"b/"+p)))
