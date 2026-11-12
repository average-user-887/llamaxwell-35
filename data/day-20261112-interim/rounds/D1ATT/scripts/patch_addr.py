import sys
p=sys.argv[1]
s=open(p).read()
a=s.index('template <bool PFK>\n__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_kernel')
b=s.index('// grid (nkvh*8 q-heads, ns), block 256')
k=s[a:b]
k=k.replace('template <bool PFK>\n__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_kernel(const ofa_params p) {',
 '// D1-ATT (12 Nov 2026), GGML_CUDA_OCCAMY_FA_ADDR=1 with PIPE=3: same kernel as ofa_decode_hs_kernel<false> but the K/V/mask\n'
 '// row addresses are strength-reduced (pointer bumps) instead of a 64-bit row*nb1 multiply per row per step, which the\n'
 '// 128-reg cap made ptxas rematerialise every iteration (~200 XMAD/IADD of ~700 loop instructions). Same loads, same FP ops,\n'
 '// same order -> bitwise-identical output.\n'
 '__global__ void __launch_bounds__(2*OFA_NT, 2) ofa_decode_hs_ar_kernel(const ofa_params p) {')
i0=k.index('    uint4 kn[OFA_T];')
m='        const float mk = __half2float(Mb[t0 + jl]);\n'
i1=k.index(m)+len(m)
new_loop='''    const size_t nbk1 = p.nbk1, nbv1 = p.nbv1;
    const char * kp = Kb + (size_t)(u0 + warp)*OFA_T*nbk1;
    const char * vp = Vb + (size_t)(u0 + warp)*OFA_T*nbv1;
    const half * mp = Mb + (u0 + warp)*OFA_T + jl;
    const size_t kstep = (size_t) OFA_NW*OFA_T*nbk1;
    const size_t vstep = (size_t) OFA_NW*OFA_T*nbv1;
    for (int u = u0 + warp; u < u1; u += OFA_NW, kp += kstep, vp += vstep, mp += OFA_NW*OFA_T) {
        uint4 kr[OFA_T], vr[OFA_T];
        {
            const char * r = kp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbk1) kr[j] = __ldg((const uint4 *) r);
        }
        {
            const char * r = vp;
#pragma unroll
            for (int j = 0; j < OFA_T; ++j, r += nbv1) vr[j] = __ldg((const uint4 *) r);
        }
        const float mk = __half2float(*mp);
'''
k=k[:i0]+new_loop+k[i1:]
body=k.split('ofa_decode_hs_ar_kernel(const ofa_params p) {',1)[1]
assert 't0' not in body and 'kn[' not in body and 'PFK' not in body, [l for l in body.split('\n') if 't0' in l or 'kn[' in l or 'PFK' in l]
s=s[:b]+k+s[b:]
def rep(o,n):
    global s
    assert s.count(o)==1, o
    s=s.replace(o,n)
rep('''    static int pipe = 0;
    if (resident == 0) {''','''    static int pipe = 0;
    static int addr = 0;
    if (resident == 0) {''')
rep('''        if (pipe < 0 || pipe > 4) pipe = 0;
''','''        if (pipe < 0 || pipe > 4) pipe = 0;
        const char * sa = getenv("GGML_CUDA_OCCAMY_FA_ADDR");
        addr = (sa && atoi(sa) != 0 && pipe == 3) ? 1 : 0;
''')
rep('''                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbp,
                    pipe == 3 ? ofa_decode_hs_kernel<false> : ofa_decode_hs_kernel<true>, 2*OFA_NT, 0));
            }''','''                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbp,
                    pipe == 3 ? ofa_decode_hs_kernel<false> : ofa_decode_hs_kernel<true>, 2*OFA_NT, 0));
                if (addr) {
                    int nba = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nba, ofa_decode_hs_ar_kernel, 2*OFA_NT, 0));
                    fprintf(stderr, "ggml_cuda: occamy FA addr mode 1 (blocks/SM %d)\\n", nba);
                    if (nba < nb) addr = 0;
                }
            }''')
rep('''        case 3:  ofa_decode_hs_kernel<false><<<''','''        case 3:  if (addr) { ofa_decode_hs_ar_kernel<<<dim3(nsplit, nkvh, ns), 2*OFA_NT, 0, stream>>>(p); break; }
                 ofa_decode_hs_kernel<false><<<''')
open(p,'w').write(s)
