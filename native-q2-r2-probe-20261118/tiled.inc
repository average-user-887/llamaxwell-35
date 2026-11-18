// Four consecutive original blocks per CTA. Warp w handles probe w in each block.
__global__ void __launch_bounds__(128) ofa_sparse_probe_tile4(
        ofa_params p, ofa_sparse_meta * metas, float * scores, int n_blocks) {
    constexpr int tile=4;
    const int lane=threadIdx.x & 31, warp=threadIdx.x>>5;
    const int plan=blockIdx.y, row=plan/p.nkvh, kvh=plan%p.nkvh;
    const int first=tile*blockIdx.x;
    const half * mask=(const half *)(p.mask+size_t(row)*p.nbm1);
    __shared__ int ends[tile][4], invalids[tile][4], bad_scores[tile][4];
    __shared__ float maxscores[tile][4];
    float kval[tile][8];
#pragma unroll
    for (int b=0;b<tile;++b) {
        const int begin=(first+b)*OFS_BLOCK;
        int end=0, bad=0;
#pragma unroll
        for (int j=0;j<2;++j) {
            const int t=begin+j*128+threadIdx.x;
            const float m=t<p.n_kv ? __half2float(mask[t]) : -INFINITY;
            const unsigned fin=__ballot_sync(0xffffffffu,isfinite(m));
            if (fin) end=begin+j*128+warp*32+32-__clz(fin);
            bad |= __any_sync(0xffffffffu,!isfinite(m) && m!=-INFINITY);
        }
        if (lane==0) { ends[b][warp]=end; invalids[b][warp]=bad; }
        const int t=begin+warp*(OFS_BLOCK/4);
        const uint4 k=t<p.n_kv ? __ldg((const uint4 *)(p.K+size_t(kvh)*p.nbk2+size_t(t)*p.nbk1+lane*16)) : make_uint4(0,0,0,0);
        ofa_cvt8(k,kval[b]);
    }
    float partial[tile][8];
#pragma unroll
    for (int h=0;h<8;++h) {
        const float4 * addr=(const float4 *)(p.Q+size_t(row)*p.nbq1+size_t(kvh*OFA_G+h)*p.nbq2)+2*lane;
        const float4 a=__ldg(addr), z=__ldg(addr+1);
        const float q[8]={a.x,a.y,a.z,a.w,z.x,z.y,z.z,z.w};
#pragma unroll
        for (int b=0;b<tile;++b) {
            float v=q[0]*kval[b][0];
#pragma unroll
            for (int i=1;i<8;++i) v=fmaf(q[i],kval[b][i],v);
            partial[b][h]=v;
        }
    }
#pragma unroll
    for (int b=0;b<tile;++b) {
        float * h=partial[b];
        ofa_rs_step_x<4,16>(h,lane);
        ofa_rs_step_x<2,8>(h,lane);
        ofa_rs_step_x<1,4>(h,lane);
        float dot=h[0]+__shfl_xor_sync(0xffffffffu,h[0],2,32);
        dot+=__shfl_xor_sync(0xffffffffu,dot,1,32);
        const int t=(first+b)*OFS_BLOCK+warp*(OFS_BLOCK/4);
        const float value=t<p.n_kv ? dot*p.scale_log2+__half2float(mask[t])*1.4426950408889634f : -INFINITY;
        const int invalid=__any_sync(0xffffffffu,!isfinite(value) && value!=-INFINITY);
        float best=fmaxf(-INFINITY,value);
#pragma unroll
        for (int d=16;d>=4;d/=2) best=fmaxf(best,__shfl_xor_sync(0xffffffffu,best,d,32));
        if (lane==0) { maxscores[b][warp]=best; bad_scores[b][warp]=invalid; }
    }
    __syncthreads();
    if (threadIdx.x<tile && first+threadIdx.x<n_blocks) {
        const int b=threadIdx.x;
        float best=-INFINITY;
        int end=0, bm=0, bs=0;
#pragma unroll
        for (int w=0;w<4;++w) {
            end=max(end,ends[b][w]); bm|=invalids[b][w]; bs|=bad_scores[b][w];
            best=fmaxf(best,maxscores[b][w]);
        }
        const size_t index=size_t(plan)*n_blocks+first+b;
        scores[index]=best;
        metas[index]={end,bm,bs};
    }
}
