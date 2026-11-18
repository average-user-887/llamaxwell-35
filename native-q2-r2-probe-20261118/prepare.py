from pathlib import Path
import difflib
here=Path(__file__).resolve().parent
base=here.parent/'last-ms-r2-selector'
f='fattn-occamy-sparse.cuh'
old=(base/f).read_text()
def one(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)
pos='static __device__ __forceinline__ bool ofa_sparse_plan_valid'
s=one(old,pos,(here/'tiled.inc').read_text()+'\n'+pos)
pos='    constexpr int NU = 4, NG = 2*NQ, NT = 32*NU*NG;'
s=one(s,pos,'''    static const bool want_probe=[] {
        const char * f=getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4");
        return f && strcmp(f,"1")==0;
    }();
    const bool tile4=revision2 && use_top16 && want_probe;
'''+pos)
line='        ofa_sparse_scan_probe_r2<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, metas.get(), scores.get(), n_blocks);'
s=one(s,line,'''        if (tile4) {
            ofa_sparse_probe_tile4<<<dim3((n_blocks+3)/4,nplans),128,0,stream>>>(p,metas.get(),scores.get(),n_blocks);
            if(nc==1) fprintf(stderr,"q2_r2_probe4_dispatch width=%d plans=%d blocks=%d tiles=%d\\n",n_kv,nplans,n_blocks,(n_blocks+3)/4);
        } else {
    '''+line+'''
        }''')
a='            copy(a, plans.get()); copy(b, old_plans.get());'
s=one(s,a,'''            std::vector<ofa_sparse_meta> actual_meta(sa.size()), reference_meta(sa.size());
            copy(actual_meta,metas.get()); copy(reference_meta,old_metas.get());
'''+a)
a='            int plan_diff = 0, score_diff = 0, output_diff = 0, nonfinite = 0;'
s=one(s,a,'''            if (tile4) {
                const bool identical=memcmp(actual_meta.data(),reference_meta.data(),actual_meta.size()*sizeof(ofa_sparse_meta))==0;
                fprintf(stderr,"q2_r2_probe4_meta entries=%zu exact=%d\\n",actual_meta.size(),int(identical));
                GGML_ASSERT(identical);
            }
'''+a)
a='            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && output_diff == 0 && nonfinite == 0);'
s=one(s,a,a+'\n'+(here/'edge.inc').read_text())
(here/f).write_text(s)
p='ggml/src/ggml-cuda/'+f
(here/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),'a/'+p,'b/'+p)))
s=(base/'compile.sh').read_text().splitlines()
s[2]=one(s[2],'CMakeFiles/ggml-cuda.dir/fattn-occamy.cu.o','/fast/fattn-occamy.cu.o')
s[5]=one(s[5],'CMakeFiles/ggml-cuda.dir/fattn-occamy.cu.o','/fast/fattn-occamy.cu.o')
(here/'compile.sh').write_text('\n'.join(s)+'\n')
