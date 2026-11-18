    // Both normalized rows fit ahead of the original projection staging region.
    {
        constexpr int slab = H/64*CH;
        static_assert(2*slab <= (TOPK*FF+FFS)/64*CH, "disjoint activation storage");
        static_assert((TOPK*FF+FFS)/64*CH+2*(H/32) <= SMEM_FLOATS, "disjoint group sums");
        float *peer_x = xs + slab;
        load_norm3(xs, sums, P.h, P.norm, red, P.eps);
        load_norm3(peer_x, sums+H/64, P.h+(1-2*row)*H, P.norm, red, P.eps);
        for (int t=physical*NW3+warp; t<NTA; t+=gridDim.x*gridDim.y*NW3) {
            float4 cw[4];
            ldA(L, t, lane, cw);
            const int expert=t>>2, quarter=t&3;
            float own=0.f, peer=0.f;
#pragma unroll
            for (int i=0;i<4;++i) {
                const int e=quarter*(QK/4)+lane+32*i;
                const int offset=((4*e)>>6)*CH+((4*e)&63);
                const float4 a=*(const float4*)(xs+offset);
                const float4 b=*(const float4*)(peer_x+offset);
                own=fmaf(cw[i].x,a.x,own); own=fmaf(cw[i].y,a.y,own);
                own=fmaf(cw[i].z,a.z,own); own=fmaf(cw[i].w,a.w,own);
                peer=fmaf(cw[i].x,b.x,peer); peer=fmaf(cw[i].y,b.y,peer);
                peer=fmaf(cw[i].z,b.z,peer); peer=fmaf(cw[i].w,b.w,peer);
            }
            own=wsum(own); peer=wsum(peer);
            if (lane==0) {
                Scratch *other=s+1-2*row;
                if (expert<NE) { s->lpart[quarter][expert]=own; other->lpart[quarter][expert]=peer; }
                else { s->gpart[quarter]=own; other->gpart[quarter]=peer; }
            }
        }
    }
    router_barrier(s-row+2);
