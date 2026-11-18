import hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
source=root.parent/'night-gdn-t2-q8-fast/occamy-gdn.cu'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='cdab6a823ca92a648f83dadf35379151347bdafac564949d0c759d36827728e9'
s=source.read_text()
def once(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
once('if constexpr (T2 == 2) {   // ---- phase A ----','if constexpr (T2 == 2 || T2 == 3) {   // ---- phase A ----')
start=s.index('        uint4 c0, c1, c2;',s.index('if constexpr (T2 == 2 || T2 == 3)'))
end=s.index('    } else {   // ---- phase A ----',start)
old=s[start:end]
loop=old[old.index('            float a0, a1;'):]
loop=loop.replace('            __syncwarp();\n','')
candidate='''        if constexpr (T2 == 3) {
            uint4 next0{}, next1{};
            unsigned next8=0;
            int t=gw;
            const unsigned * p=(const unsigned *)L.qkv + size_t(t)*(ROW_H/4);
            if (t<NROWS_A) {
                next0=__ldg((const uint4 *)p+lane);
                next1=__ldg((const uint4 *)p+32+lane);
                next8=__ldg(p+256+lane);
            }
            for (; t<NROWS_A; t+=W) {
                const unsigned w[9]={next0.x,next0.y,next0.z,next0.w,next1.x,next1.y,next1.z,next1.w,next8};
                if (t+W<NROWS_A) {
                    p+=size_t(W)*(ROW_H/4);
                    next0=__ldg((const uint4 *)p+lane);
                    next1=__ldg((const uint4 *)p+32+lane);
                    next8=__ldg(p+256+lane);
                }
''' + loop + '        } else {\n' + old + '        }\n'
s=s[:start]+candidate+s[end:]
anchor='// GDN2: MTP1 verify window (one sequence, two tokens) as one gdn_fused2<1> launch.'
once(anchor,(root/'packed-host.inc').read_text()+'\n'+anchor)
anchor='    if (events) n82_before(ctx.device, st, m.wqkv->name, d.gridT2, d.q8a);'
once(anchor,'''    static const bool packed = env_flag("GGML_CUDA_OCCAMY_GDN_T2_Q8A_PACKED");
    if (packed) {
        GGML_ASSERT(d.q8a && d.gridT2==72);
        static thread_local bool checked=false;
        if (!checked) {
            int active=0;
            cudaFuncAttributes f{};
            CUDA_CHECK(cudaFuncGetAttributes(&f,gdn_fused2<3>));
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,gdn_fused2<3>,NT,SMEM3_FLOATS*sizeof(float)));
            GGML_ASSERT(active==3 && f.numRegs<=80 && f.localSizeBytes==0);
            fprintf(stderr,"Q8PA_ADMISSION grid=%d active=%d regs=%d local=%zu\\n",d.gridT2,active,f.numRegs,f.localSizeBytes);
            checked=true;
        }
        PP.p[0].L.qkv=q8pa_lookup(ctx.device,st,PP.p[0].L,m.wqkv->name,false);
    }
'''+anchor)
once('    if (d.q8a) {\n        gdn_fused2<2>', '''    if (packed) {
        gdn_fused2<3><<<d.gridT2, NT, SMEM3_FLOATS * sizeof(float), st>>>(PP);
    } else if (d.q8a) {
        gdn_fused2<2>''')
once('    bool candidate = false;', '    int specialization = 0;')
once('int grid, bool candidate) {', 'int grid, int specialization) {')
once('e.candidate = candidate;', 'e.specialization = specialization;')
once('e.candidate == candidate', 'e.specialization == specialization')
once('e.candidate ? 2 : 1, i,', 'e.specialization, i,')
once('m.wqkv->name, d.gridT2, d.q8a);', 'm.wqkv->name, d.gridT2, packed ? 3 : (d.q8a ? 2 : 1));')
(root/'occamy-gdn.cu').write_text(s)
