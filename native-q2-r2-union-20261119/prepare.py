from pathlib import Path
import difflib
c=Path(__file__).resolve().parent
base=c.parent/'finish33-r2-stride'
old=(base/'fattn-occamy-sparse.cuh').read_text()
def rep(s,k,v):
    assert s.count(k)==1,k
    return s.replace(k,v)
s=old
i=s.index('template <unsigned TOKEN_BYTES>\n__global__')
j=s.index('static std::atomic<long long> ofa_sparse_calls',i)
fallback=s[i:j].replace('ofa_sparse_attention_q2_stride','ofa_sparse_union_fallback')
fallback=rep(fallback,'    constexpr int NQ = 2, NU = 4;', '''    __shared__ int eligible;
    if (threadIdx.x==0) eligible=ofa_union_eligible(plans,blockIdx.y,p.n_kv);
    __syncthreads();
    if (eligible) return;
    constexpr int NQ = 2, NU = 4;''')
fallback=rep(fallback,'(i3*p.nkvh + kvh)*p.nsplit + split','(i3*p.nkvh + kvh)*24 + split')
src=(c.parent/'sparse/r2/src/ggml/src/ggml-cuda/fattn-occamy.cu').read_text()
start=src.index('__global__ void __launch_bounds__(OFA_D) ofa_mrow_combine_kernel')
combine=src[start:src.index('\n}',start)+2]
combine=combine.replace('ofa_mrow_combine_kernel','ofa_union_combine').replace('const int nsplit,','const ofa_sparse_plan * plans, const int n_kv,')
combine=rep(combine,'const int blk0 = (i3*nkvh + kvh)*nsplit;', '''__shared__ int nsplit;
    if (threadIdx.x==0) nsplit=ofa_union_eligible(plans,kvh,n_kv) ? 24 : 12;
    __syncthreads();
    const int blk0=(i3*nkvh+kvh)*24;''')
helper='''
static void ofa_union_retain(const ofa_params &p,ofa_sparse_plan *plans,cudaStream_t stream) {
    if (p.nbk1==1024) {
        ofa_sparse_union_q2<1024><<<dim3(24,2),128,0,stream>>>(p,plans);
        ofa_sparse_union_fallback<1024><<<dim3(12,2),512,0,stream>>>(p,plans);
    } else {
        ofa_sparse_union_q2<2048><<<dim3(24,2),128,0,stream>>>(p,plans);
        ofa_sparse_union_fallback<2048><<<dim3(12,2),512,0,stream>>>(p,plans);
    }
}
static void ofa_union_merge(const ofa_params &p,ofa_sparse_plan *plans,char *out,const ggml_tensor *dst,cudaStream_t stream) {
    ofa_union_combine<<<dim3(16,2),256,0,stream>>>(p.part_o,p.part_ml,out,plans,p.n_kv,2,2,dst->nb[1],dst->nb[2],dst->nb[3]);
}
static void ofa_union_launch(const ofa_params &p,ofa_sparse_plan *plans,char *out,const ggml_tensor *dst,cudaStream_t stream) {
    ofa_union_retain(p,plans,stream);
    ofa_union_merge(p,plans,out,dst,stream);
}
'''
s=s[:j]+(c/'union.inc').read_text()+'\n'+fallback+combine+'\n'+helper+s[j:]
seg=s.index('static void ofa_sparse_launch')
prefix,s=s[:seg],s[seg:]
s=rep(s,'    const size_t nblk = (size_t) nsplit*nkvh;', '''    static const bool want_union=[] {
        const char *s=getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_UNION"); return s && strcmp(s,"1")==0;
    }();
    static const bool union_stride=[] {
        const char *s=getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE"); return s && strcmp(s,"1")==0;
    }();
    const bool union_mode=want_union && union_stride && tile4 && nsplit==12 && blocks_per_sm==1 &&
        k->nb[2]==512 && v->nb[2]==512 && k->nb[1]==v->nb[1] && (k->nb[1]==1024 || k->nb[1]==2048);
    const size_t nblk = (size_t) nsplit*nkvh*(union_mode?2:1);''')
s=rep(s,'    if (stride) {\n        static std::atomic<int> stride_seen', '''    if (union_mode) {
        static std::atomic<long long> calls{0};
        const long long n=++calls;
        if (n<=1 || n%1000==0) fprintf(stderr,"q2_union_dispatch calls=%lld width=%d pitch=%zu split=24 threads=128 tile=16 fallback_split=12\\n",n,n_kv,p.nbk1);
        ofa_union_retain(p,plans.get(),stream);
    } else if (stride) {
        static std::atomic<int> stride_seen''')
anchor='    ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, NQ, 1), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), (char *) dst->data'
s=rep(s,anchor,'''    if (union_mode) ofa_union_merge(p,plans.get(),(char *)dst->data,dst,stream);
    else '''+anchor.strip())
s=rep(s,'                plan_diff += memcmp(&a[i], &b[i], bytes) != 0;', '''                if (union_mode) {
                    plan_diff+=memcmp(&a[i],&b[i],offsetof(ofa_sparse_plan,sparse_warps))!=0 ||
                        memcmp(a[i].blocks,b[i].blocks,b[i].n_blocks*sizeof(int))!=0;
                } else plan_diff += memcmp(&a[i], &b[i], bytes) != 0;''')
s=rep(s,'            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && output_diff == 0 && nonfinite == 0);',
      '            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && (union_mode || output_diff == 0) && nonfinite == 0);\n'+(c/'check.inc').read_text())
s=rep(s,'if (stride && p.nbk1 == 2048','if (!union_mode && stride && p.nbk1 == 2048')
s=prefix+s
(c/'fattn-occamy-sparse.cuh').write_text(s)
(c/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),'a/ggml/src/ggml-cuda/fattn-occamy-sparse.cuh','b/ggml/src/ggml-cuda/fattn-occamy-sparse.cuh')))
