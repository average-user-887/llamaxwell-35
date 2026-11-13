#pragma once
// Q8N (13 Nov 2026): lane load/realign/scale math for the native q8_0 KV read in the occamy hs_ar decode kernel.
// Shared by the CUDA kernel and the host unit check (tests/occamy-kvq8-lane-check.cpp), so the check runs this exact code.
// One q8_0 row of 256 dims = 8 blocks of 34 B {f16 d; int8 qs[32]} = 272 B; the row start is 4-B aligned.
// Lane l owns dims [8l, 8l+8): block b = l>>2, quarter q = l&3, qs bytes at o = 34b + 2 + 8q.
//   even b: o%4 == 2 -> 3 aligned words at (o & ~3), realign by 2 bytes (byte_perm 0x5432)
//   odd  b: o%4 == 0 -> 2 aligned words, the third load is clamped onto word 1 (no read past the row), perm 0x3210
// Scale d: one aligned u16 at 34b (offset doff from the lane base, may be negative).
#include <cstdint>

#if defined(__CUDACC__)
#define OFQ_HD static __host__ __device__ __forceinline__
#else
#define OFQ_HD static inline
#endif

struct ofq_lane {
    int      base;   // byte offset of word 0 from the row start (4-B aligned)
    int      w2off;  // byte offset of word 2 from base (8, or 4 = clamped onto word 1)
    int      doff;   // byte offset of the f16 scale from base
    uint32_t sel;    // byte_perm selector
};

OFQ_HD ofq_lane ofq_lane_geom(const int lane) {
    const int b = lane >> 2;
    const int q = lane & 3;
    const int o = 34*b + 2 + 8*q;
    ofq_lane g;
    g.base  = o & ~3;
    g.w2off = (o & 2) ? 8 : 4;
    g.doff  = 34*b - g.base;
    g.sel   = (o & 2) ? 0x5432u : 0x3210u;
    return g;
}

OFQ_HD uint32_t ofq_perm(const uint32_t x, const uint32_t y, const uint32_t s) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, y, s);
#else
    // PRMT default mode: result byte n = byte (s >> 4n) & 7 of the 8-byte value {y:x}
    const uint64_t v = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) {
        r |= (uint32_t) ((v >> (8*((s >> (4*n)) & 7))) & 0xFF) << (8*n);
    }
    return r;
#endif
}

// realign the 3 loaded words into the lane's 8 int8 values (lo = dims 0..3, hi = dims 4..7)
OFQ_HD void ofq_realign(const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t sel, uint32_t & lo, uint32_t & hi) {
    lo = ofq_perm(w0, w1, sel);
    hi = ofq_perm(w1, w2, sel);
}

OFQ_HD void ofq_unpack8(const uint32_t lo, const uint32_t hi, float * f) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[i]     = (float) (int8_t) (lo >> (8*i));
        f[4 + i] = (float) (int8_t) (hi >> (8*i));
    }
}
