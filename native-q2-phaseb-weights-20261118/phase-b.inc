    // Both rows have published Phase-A partials; every CTA owns one row for A and C.
    // During B a warp pair cooperates only on projections whose weight row is identical.
    __shared__ int peer_sel[TOPK];
    __shared__ float peer_w[TOPK];
    __shared__ int common_rank0[TOPK + 1], common_rank1[TOPK + 1];
    __shared__ int solo_rank[TOPK], common_count;
    Scratch * const both = s - row;
    float * const peer_x = xs + (H / 64) * CH;
    float2 * const peer_sum = sums + H / 64;
    if (warp == 0) route(s, lane, sel, selw);
    if (warp == 1) route(both + (1-row), lane, peer_sel, peer_w);
    load_norm3(peer_x, peer_sum, P.h + (1-2*row)*H, P.norm, red, P.eps);
    // load_norm3 ends with a block barrier. Match actual expert IDs, irrespective of rank.
    if (capture_ids && local_block == 0 && threadIdx.x < TOPK)
        s->logits[threadIdx.x] = float(sel[threadIdx.x]);
    if (warp == 0) {
        const int * const ranks0 = row ? peer_sel : sel;
        const int * const ranks1 = row ? sel : peer_sel;
        const int rank = lane & (TOPK - 1);
        int mate = -1, own_mate = -1;
#pragma unroll
        for (int j=0; j<TOPK; ++j) {
            if (ranks0[rank] == ranks1[j]) mate = j;
            if (sel[rank] == peer_sel[j]) own_mate = j;
        }
        const unsigned shared_mask = __ballot_sync(0xffffffffu, lane < TOPK && mate >= 0);
        const unsigned solo_mask = __ballot_sync(0xffffffffu, lane < TOPK && own_mate < 0);
        if (lane < TOPK && mate >= 0) {
            const int group = 1 + __popc(shared_mask & ((1u << rank)-1));
            common_rank0[group] = rank;
            common_rank1[group] = mate;
        }
        if (lane < TOPK && own_mate < 0)
            solo_rank[__popc(solo_mask & ((1u << rank)-1))] = rank;
        if (lane == 0) {
            common_rank0[0] = common_rank1[0] = -1;
            common_count = 1 + __popc(shared_mask);
        }
    }
    __syncthreads();
    {
        const int common = common_count;
        const int partner = warp & 1; // absolute token row 0 or 1
        const float * const xc = (partner == row ? xs : peer_x) + lane * CH;
        const float2 sx = (partner == row ? sums : peer_sum)[lane];
        unsigned * const gate_tile = (unsigned *)(smem + SMEM_FLOATS) + (warp & ~1)*STW3;
        unsigned * const up_tile = gate_tile + STW3;
        // Each block owns eight adjacent output channels per pass. The end of each
        // expert is aligned to eight channels, so every CTA barrier is block-uniform.
        for (int first=physical*(NW3/2); first<common*FF; first+=48*(NW3/2)) {
            const int t=first+(warp>>1), group=t/FF, channel=t%FF;
            const int rank0=common_rank0[group], rank1=common_rank1[group];
            const uint8_t * weights;
            if (group == 0) {
                weights = (partner ? L.up_sh : L.gate_sh) + size_t(channel)*ROW_GU;
            } else {
                const int expert=(row ? peer_sel : sel)[rank0];
                weights = (partner ? L.up_exps : L.gate_exps) + (size_t(expert)*FF+channel)*ROW_GU;
            }
            uint4 chunk[3];
            ld72(weights, lane, chunk);
            st72(stg4, lane, chunk);
            __syncthreads();
            const float gg=wsum(dot2c<true>(lds9(gate_tile+lane*9), xc, sx));
            const float uu=wsum(dot2c<true>(lds9(up_tile+lane*9), xc, sx));
            if (lane == 0) {
                Scratch * dst=both+partner;
                const float y=silu(gg)*uu;
                if (group == 0) dst->h_sh[channel]=y;
                else dst->h_exp[(partner ? rank1 : rank0)*FF + channel]=y;
            }
            __syncthreads();
        }
        // This list is empty under full routed overlap and contains all eight ranks
        // under disjoint routing. Every solo weight is consumed by its own row.
        const int end=(TOPK+1-common)*FF;
        const float * const own_x=xs+lane*CH;
        const float2 own_sum=sums[lane];
        uint4 cv[3]; const uint8_t *g, *u;
        if (gw < end) {
            int t=gw;
            int dest=solo_rank[t/FF]*FF+t%FF;
            rowB(L,sel,FFS+dest,g,u); ld72(g,lane,cv);
            for (;;) {
                st72(stg4,lane,cv); __syncwarp();
                ld72(u,lane,cv);
                const float gg=wsum(dot2c<true>(lds9(stg+lane*9),own_x,own_sum));
                __syncwarp(); st72(stg4,lane,cv); __syncwarp();
                const int next=t+W;
                int next_dest=0;
                if (next<end) {
                    next_dest=solo_rank[next/FF]*FF+next%FF;
                    rowB(L,sel,FFS+next_dest,g,u); ld72(g,lane,cv);
                }
                const float uu=wsum(dot2c<true>(lds9(stg+lane*9),own_x,own_sum));
                if (lane==0) s->h_exp[dest]=silu(gg)*uu;
                __syncwarp();
                if (next>=end) break;
                t=next; dest=next_dest;
            }
        }
    }
