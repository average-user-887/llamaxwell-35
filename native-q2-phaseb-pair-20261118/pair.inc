// Each lane retains its original two 32-element accumulation chains for each projection.
__device__ __forceinline__ void add_four(float & a, unsigned packed, float4 x) {
    a = fmaf(nib(packed,0),x.x,a); a = fmaf(nib(packed,1),x.y,a);
    a = fmaf(nib(packed,2),x.z,a); a = fmaf(nib(packed,3),x.w,a);
}
__device__ __forceinline__ float2 gate_up_pair(const Q9 & g,const Q9 & u,const float * xc,float2 sx) {
    float ag0=0.f, ag1=0.f, au0=0.f, au1=0.f;
#pragma unroll
    for (int i=0; i<4; ++i) {
        const unsigned g0=__byte_perm(g.w[i],g.w[i+1],0x5432);
        const unsigned u0=__byte_perm(u.w[i],u.w[i+1],0x5432);
        const float4 x0=*(const float4 *)(xc+4*i), x1=*(const float4 *)(xc+16+4*i);
        add_four(ag0,unpk(g0),x0); add_four(au0,unpk(u0),x0);
        add_four(ag0,unpk(g0>>4),x1); add_four(au0,unpk(u0>>4),x1);
        const unsigned g1=g.w[5+i], u1=u.w[5+i];
        const float4 x2=*(const float4 *)(xc+32+4*i), x3=*(const float4 *)(xc+48+4*i);
        add_four(ag1,unpk(g1),x2); add_four(au1,unpk(u1),x2);
        add_four(ag1,unpk(g1>>4),x3); add_four(au1,unpk(u1>>4),x3);
    }
    const float gd0=__half2float(__ushort_as_half((unsigned short)g.w[0]));
    const float gd1=__half2float(__ushort_as_half((unsigned short)(g.w[4]>>16)));
    const float ud0=__half2float(__ushort_as_half((unsigned short)u.w[0]));
    const float ud1=__half2float(__ushort_as_half((unsigned short)(u.w[4]>>16)));
    return make_float2(gd0*(128.f*ag0-136.f*sx.x)+gd1*(128.f*ag1-136.f*sx.y),
                       ud0*(128.f*au0-136.f*sx.x)+ud1*(128.f*au1-136.f*sx.y));
}
