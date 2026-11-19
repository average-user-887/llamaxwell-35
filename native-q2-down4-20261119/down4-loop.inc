    const float gs = 1.f / (1.f + expf(-gl));
    const int group = lane & 7;
    // r and H are multiples of four, so all four rows are always valid.
    for (; r < H; r += 4 * W) {
        float even = 0.f, odd = 0.f;
#pragma unroll 1
        for (int k = 0; k < 4; ++k) {
            st72(stg4, lane, cv);
            __syncwarp();
            ld_down4(L, sel, r, k + 4, lane, cv);
            const int ua = k * 8 + group;
            const float dA = dot2c<true, true>(lds9(stg + lane * 9), xs + ua * CH, sums[ua]);
            __syncwarp();
            st72(stg4, lane, cv);
            __syncwarp();
            if (k == 0) {
                ld_down4(L, sel, r, TOPK, lane, cv);
            } else if (k < 3) {
                ld_down4(L, sel, r, k + 1, lane, cv);
            } else if (r + 4 * W < H) {
                ld_down4(L, sel, r + 4 * W, 0, lane, cv);
            }
            const int ub = ua + 32;
            const float dB = dot2c<true, true>(lds9(stg + lane * 9), xs + ub * CH, sums[ub]);
            float value = __fmaf_rn(selw[k], dA, __fmul_rn(selw[k + 4], dB));
            __syncwarp();
            if (k == 0) {
                st72(stg4, lane, cv);
                __syncwarp();
                ld_down4(L, sel, r, 1, lane, cv);
                const int us = group + 64;
                const float shared = dot2c<true, true>(lds9(stg + lane * 9), xs + us * CH, sums[us]);
                even = __fmaf_rn(gs, shared, value);
                __syncwarp();
            } else if (k == 1) {
                odd = value;
            } else if (k == 2) {
                even = __fadd_rn(even, value);
            } else {
                odd = __fadd_rn(odd, value);
            }
        }
        // Lane zero of the old 32-way butterfly first combined virtual lanes
        // (0,16) and (8,24), then those two sums. Finish inside each row's octet.
        float acc = __fadd_rn(even, odd);
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o, 8);
        if (group == 0) P.out[r + (lane >> 3)] = P.h[r + (lane >> 3)] + acc;
    }
}
