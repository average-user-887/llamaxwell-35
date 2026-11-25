// Q8N host unit check: the exact lane load/realign/scale math of ofa_decode_hs_ar_q8_kernel (fattn-occamy-kvq8.cuh)
// against an fp32 reference dequant of q8_0 rows. Build: g++ -O2 -std=c++17 -I ggml/src/ggml-cuda tests/occamy-kvq8-lane-check.cpp
#include "fattn-occamy-kvq8.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct blk { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(blk) == 34, "q8_0 block");

static float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float) h; }
static uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

int main() {
    const int D = 256, NB = D/32, ROW = NB*34, NH = 2, NTOK = 4096;
    const size_t nb2 = ROW, nb1 = (size_t) NH*ROW;
    std::mt19937 rng(20261113);
    std::uniform_int_distribution<int> i8(-128, 127), kind(0, 9), pos(0, 31);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    long n_rows = 0, n_vals = 0, n_bad = 0, n_zero_d = 0, n_m128 = 0, n_p127 = 0, n_align = 0;
    double max_kerr = 0.0, max_verr = 0.0;
    for (int base_off = 0; base_off < 16; base_off += 4) {   // data % 4 == 0 is all the kernel requires
        std::vector<uint8_t> buf(nb1*NTOK + 64, 0xCD);
        uint8_t * data = buf.data() + base_off;
        // fill
        for (int t = 0; t < NTOK; ++t) for (int h = 0; h < NH; ++h) {
            uint8_t * row = data + t*nb1 + h*nb2;
            for (int b = 0; b < NB; ++b) {
                blk x;
                const int k = kind(rng);
                if (k == 0) {                       // scale 0 block
                    x.d = 0; for (int i = 0; i < 32; ++i) x.qs[i] = (int8_t) i8(rng);
                } else if (k <= 2) {                // raw full-range int8, forced extremes, random finite scale
                    x.d = f2h(std::ldexp(1.0f + 0.5f*nd(rng)*0.1f, (int)(rng() % 20) - 14));
                    for (int i = 0; i < 32; ++i) x.qs[i] = (int8_t) i8(rng);
                    x.qs[pos(rng)] = -128; x.qs[pos(rng)] = 127;
                } else {                            // ggml-style quantised gaussian row (d = amax/127)
                    float v[32], amax = 0.0f;
                    const float sc = std::exp(nd(rng));
                    for (int i = 0; i < 32; ++i) { v[i] = nd(rng)*sc; amax = std::fmax(amax, std::fabs(v[i])); }
                    const float d = amax/127.0f, id = d ? 1.0f/d : 0.0f;
                    x.d = f2h(d);
                    for (int i = 0; i < 32; ++i) x.qs[i] = (int8_t) std::lround(v[i]*id);
                }
                memcpy(row + 34*b, &x, 34);
            }
        }
        for (int t = 0; t < NTOK; ++t) for (int h = 0; h < NH; ++h) {
            const uint8_t * row = data + t*nb1 + h*nb2;
            ++n_rows;
            // fp32 reference dequant
            float ref[256];
            for (int b = 0; b < NB; ++b) {
                blk x; memcpy(&x, row + 34*b, 34);
                if (x.d == 0) ++n_zero_d;
                for (int i = 0; i < 32; ++i) {
                    ref[32*b + i] = h2f(x.d)*(float) x.qs[i];
                    n_m128 += x.qs[i] == -128; n_p127 += x.qs[i] == 127;
                }
            }
            for (int lane = 0; lane < 32; ++lane) {
                const ofq_lane g = ofq_lane_geom(lane);
                const uint8_t * r = row + g.base;
                const uint8_t * a[3] = { r, r + 4, r + g.w2off };
                uint32_t w[3];
                for (int i = 0; i < 3; ++i) {
                    if (((uintptr_t) a[i] & 3) || a[i] < row || a[i] + 4 > row + ROW) ++n_align;
                    memcpy(&w[i], a[i], 4);
                }
                const uint8_t * ad = r + g.doff;
                if (((uintptr_t) ad & 1) || ad != row + 34*(lane >> 2)) ++n_align;
                uint16_t du; memcpy(&du, ad, 2);
                uint32_t lo, hi;
                ofq_realign(w[0], w[1], w[2], g.sel, lo, hi);
                float f[8];
                ofq_unpack8(lo, hi, f);
                const float d = h2f(du);
                // dequant must match exactly
                for (int i = 0; i < 8; ++i) {
                    ++n_vals;
                    if (f[i]*d != ref[8*lane + i]) {
                        if (n_bad < 10) fprintf(stderr, "MISMATCH off=%d t=%d h=%d lane=%d i=%d got=%g ref=%g\n",
                                                base_off, t, h, lane, i, f[i]*d, ref[8*lane + i]);
                        ++n_bad;
                    }
                }
                // K path: (sum q*x)*d vs sum q*(d*x); V path: (p*d)*x vs p*(d*x)
                float qv[8], acc, racc = 0.0f, mag = 0.0f;
                for (int i = 0; i < 8; ++i) qv[i] = nd(rng);
                acc = qv[0]*f[0];
                for (int i = 1; i < 8; ++i) acc = std::fma(qv[i], f[i], acc);
                acc *= d;
                for (int i = 0; i < 8; ++i) { racc += qv[i]*ref[8*lane + i]; mag += std::fabs(qv[i]*ref[8*lane + i]); }
                if (mag > 0) max_kerr = std::fmax(max_kerr, std::fabs(acc - racc)/mag);
                const float pj = std::fabs(nd(rng));
                for (int i = 0; i < 8; ++i) {
                    const float vm = std::fabs(pj*ref[8*lane + i]);
                    if (vm > 0) max_verr = std::fmax(max_verr, std::fabs((pj*d)*f[i] - pj*ref[8*lane + i])/vm);
                }
            }
        }
    }
    const bool ok = n_bad == 0 && n_align == 0 && max_kerr < 1e-5 && max_verr < 1e-6 && n_zero_d > 0 && n_m128 > 0 && n_p127 > 0;
    printf("rows=%ld vals=%ld exact_mismatch=%ld align_or_bounds_violations=%ld zero_scale_blocks=%ld int8_-128=%ld int8_127=%ld "
           "max_rel_err_Kdot=%.3e max_rel_err_Vscale=%.3e base_offsets=0,4,8,12 -> %s\n",
           n_rows, n_vals, n_bad, n_align, n_zero_d, n_m128, n_p127, max_kerr, max_verr, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
