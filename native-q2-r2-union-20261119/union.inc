// Valid plans carry up to 16 old pages and 17 recent pages. On unexpected
// geometry the guarded original 12-split primitive handles both rows of a head.
static __device__ __forceinline__ bool ofa_union_eligible(const ofa_sparse_plan * plans, int head, int width) {
    for (int row=0; row<2; ++row) {
        const auto & v=plans[row*2+head];
        if (!ofa_sparse_plan_valid(v,width) || v.visible_end-v.recent_begin > OFS_RECENT+OFS_BLOCK) return false;
    }
    return true;
}

static __device__ __forceinline__ int ofa_union_page(const ofa_sparse_plan & v, int slot) {
    if (slot<v.n_blocks) return v.blocks[slot];
    const int page=v.recent_begin/OFS_BLOCK+slot-v.n_blocks;
    return page*OFS_BLOCK<v.visible_end ? page : -1;
}

static __device__ __forceinline__ bool ofa_union_in_plan(const ofa_sparse_plan & v, int page) {
    if (page>=v.recent_begin/OFS_BLOCK) return page*OFS_BLOCK<v.visible_end;
    int lo=0, hi=v.n_blocks;
    while (lo<hi) {
        const int m=(lo+hi)/2;
        if (v.blocks[m]<page) lo=m+1; else hi=m;
    }
    return lo<v.n_blocks && v.blocks[lo]==page;
}

// Four token owners in one warp share each Q coordinate read. K vectors
// stay in registers throughout the four-by-eight dot product tile.
static __device__ __forceinline__ void ofa_union_qk8(
        const float (&keys)[4][8],const float (*queries)[8][32],float (&score)[16][16],
        const half * mask,int first,int stop,int row,int warp,int lane) {
    float dots[4][8];
#pragma unroll
    for (int h=0;h<8;++h) {
        float accum[4];
        const float q0=queries[h][0][lane];
#pragma unroll
        for (int j=0;j<4;++j) accum[j]=q0*keys[j][0];
#pragma unroll
        for (int i=1;i<8;++i) {
            const float q=queries[h][i][lane];
#pragma unroll
            for (int j=0;j<4;++j) accum[j]=fmaf(q,keys[j][i],accum[j]);
        }
#pragma unroll
        for (int j=0;j<4;++j) dots[j][h]=accum[j];
    }
#pragma unroll
    for (int j=0;j<4;++j) {
        float * h=dots[j];
        ofa_rs_step_x<4,16>(h,lane);
        ofa_rs_step_x<2,8>(h,lane);
        ofa_rs_step_x<1,4>(h,lane);
        float s=h[0]+__shfl_xor_sync(0xffffffff,h[0],2);
        s+=__shfl_xor_sync(0xffffffff,s,1);
        const int pos=warp+j*4,token=first+pos;
        if ((lane&3)==0) score[pos][row*8+lane/4]=token<stop ?
            s+__half2float(mask[token])*1.4426950408889634f : -INFINITY;
    }
}

template<unsigned PITCH>
__global__ void __launch_bounds__(128) ofa_sparse_union_q2(const ofa_params p, ofa_sparse_plan * plans) {
    constexpr int TILE=16, PAGES=33, SPLITS=24;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32,kvh=blockIdx.y,split=blockIdx.x;
    __shared__ int eligible,desc[2*PAGES],end[2],count[2];
    __shared__ float q[16][8][32],score[TILE][16],running_max[16],denom[16],alpha[16];
    if (tid==0) eligible=ofa_union_eligible(plans,kvh,p.n_kv);
    __syncthreads();
    if (!eligible) return;
    if (tid<2) {
        const auto & plan=plans[2*tid+kvh];
        end[tid]=plan.visible_end;
        count[tid]=plan.n_blocks+(plan.visible_end-plan.recent_begin+255)/256;
    }
    if (tid<16) { running_max[tid]=-FLT_MAX/2; denom[tid]=0.0f; }
    if (tid<2*PAGES) {
        const int row=tid/PAGES;
        const int page=ofa_union_page(plans[2*row+kvh],tid%PAGES);
        int tagged=0;
        if (page>=0) {
            const bool shared=ofa_union_in_plan(plans[(1-row)*2+kvh],page);
            if (row==0 || !shared) tagged=(page<<2) | (shared?3:(1<<row));
        }
        desc[tid]=tagged;
    }
    for (int i=tid;i<16*256;i+=128) {
        const int head=i/256, d=i%256;
        q[head][d%8][d/8]=((const float *)(p.Q+(head/8)*p.nbq1+(kvh*8+head%8)*p.nbq2))[d]*p.scale_log2;
    }
    float out[16][2]={};
    __syncthreads();
    int participated=0;
    const int ntiles=(PAGES+count[1])*(256/TILE);
    // Row zero enumerates its pages. Row one contributes only pages absent
    // from row zero. Shared pages are visited once with independent row masks.
    for (int t=split; t<ntiles; t+=SPLITS) {
        const int tag=desc[t/(256/TILE)];
        if (!tag) continue;
        const int page=tag>>2;
        const int first=page*256+(t%(256/TILE))*TILE;
        // The original traverses complete four-key units. Only their tail
        // padding may enter the arithmetic (with the original row mask).
        const int stop0=(end[0]+3)&~3, stop1=(end[1]+3)&~3;
        const int active=((tag&1) && first<stop0 ? 1 : 0) |
                         ((tag&2) && first<stop1 ? 2 : 0);
        if (!active) continue;
        participated|=active;
        float keys[4][8];
#pragma unroll
        for (int j=0;j<4;++j) {
            const unsigned token=first+warp+j*4;
            ofa_cvt8(__ldg((const uint4 *)(p.K+token*PITCH+kvh*512u+lane*16u)),keys[j]);
        }
#pragma unroll
        for (int row=0;row<2;++row) if (active&(1<<row)) {
            const half * mask=(const half *)(p.mask+row*p.nbm1);
            ofa_union_qk8(keys,&q[row*8],score,mask,first,row?stop1:stop0,row,warp,lane);
        }
        __syncthreads();
#pragma unroll
        for (int group=0;group<4;++group) {
            const int h=warp+group*4;
            if (!(active&(1<<(h/8)))) continue;
            const int j=lane&15;
            const float s=score[j][h];
            float m=fmaxf(running_max[h],s);
#pragma unroll
            for (int d=8;d>0;d/=2) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
            const float prob=exp2f(s-m), a=exp2f(running_max[h]-m);
            float sum=prob;
#pragma unroll
            for (int d=8;d>0;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
            if (lane<16) score[j][h]=prob;
            if (lane==0) { alpha[h]=a; denom[h]=denom[h]*a+sum; running_max[h]=m; }
        }
        __syncthreads();
#pragma unroll
        for (int h=0;h<16;++h) if (active&(1<<(h/8))) {
            out[h][0]*=alpha[h]; out[h][1]*=alpha[h];
        }
#pragma unroll
        for (int j=0;j<TILE;++j) {
            const unsigned token=first+j;
            const float2 v=__half22float2(__ldg((const half2 *)(p.V+token*PITCH+kvh*512u+tid*4u)));
#pragma unroll
            for (int h=0;h<16;++h) if ((active&(1<<(h/8))) && int(token)<(h<8?stop0:stop1)) {
                const float w=score[j][h];
                out[h][0]=fmaf(w,v.x,out[h][0]); out[h][1]=fmaf(w,v.y,out[h][1]);
            }
        }
        __syncthreads();
    }
    if (tid<2 && (participated&(1<<tid))) atomicAdd(&plans[tid*2+kvh].sparse_warps,4);
    const int blk=kvh*SPLITS+split;
#pragma unroll
    for (int h=0;h<16;++h) {
        ((float2 *)(p.part_o+(blk*16+h)*256))[tid]=make_float2(out[h][0],out[h][1]);
    }
    if (tid<16) p.part_ml[blk*16+tid]=make_float2(running_max[tid],denom[tid]);
}
