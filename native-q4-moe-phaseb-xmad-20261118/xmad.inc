// A CTA has finished its FP32 routing partials. The upper region of xs is
// unused until Phase C. Put byte-expanded, paired activations there once,
// for all gate and up dot products assigned to the CTA.
constexpr int XM_PACK_FLOAT = 2304;
constexpr int XM_SCALE_FLOAT = XM_PACK_FLOAT + H/2;
static_assert((H/64)*CH <= XM_PACK_FLOAT && XM_SCALE_FLOAT + (H/32)*2 <= (TOPK*FF+FFS)/64*CH,
              "phase-B cache must fit above the normalized row and below the existing phase-C slab");
__device__ __forceinline__ void xm_prepare(const float * xs, uint4 * packed, float2 * scale) {
    const int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < H; i += NT3) {
        const int block = i >> 5;
        const float value = xs[(i >> 6)*CH + (i & 63)];
        float amax = fabsf(value);
#pragma unroll
        for (int shift=16; shift>0; shift>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,shift));
        const float sum=wsum(value);
        const float d=amax/127.f;
        const int q=amax==0.f ? 0 : int(roundf(value/d));
        const int q2=__shfl_down_sync(0xffffffffu,q,2);
        if (lane==0) scale[block]=__half22float2(make_half2(d,sum));
        if ((lane & 2)==0) {
            const int word=((block & 1)*4 + ((lane & 15)>>2))*32 + (block>>1);
            const int field=((lane>>4)*2)+(lane&1);
            ((unsigned *)packed)[4*word+field]=(unsigned(q)&65535u)|(unsigned(q2)<<16);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ float xm_dot(const unsigned * w, const uint4 * packed,
                                        const float2 * scale, int lane) {
    float accum=0.f;
#pragma unroll
    for (int b=0; b<2; ++b) {
        int even=0, odd=0;
#pragma unroll
        for (int j=0; j<4; ++j) {
            const unsigned v=b==0 ? __byte_perm(w[j],w[j+1],0x5432) : w[j+5];
            const uint4 a=packed[(b*4+j)*32+lane];
            constexpr unsigned mask=0x000f000fu;
            even=ggml_cuda_sm52_mad2(v&mask, a.x, even);
            odd=ggml_cuda_sm52_mad2((v>>8)&mask, a.y, odd);
            even=ggml_cuda_sm52_mad2((v>>4)&mask, a.z, even);
            odd=ggml_cuda_sm52_mad2((v>>12)&mask, a.w, odd);
        }
        const unsigned bits=b==0 ? w[0] : w[4]>>16;
        const float d4=__half2float(__ushort_as_half((unsigned short)bits));
        const float2 q8=scale[lane*2+b];
        accum+=d4*((even+odd)*q8.x-8.f*q8.y);
    }
    return accum;
}
