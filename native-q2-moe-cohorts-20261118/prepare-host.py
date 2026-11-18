from pathlib import Path
import difflib
base=Path('operator.base.cpp').read_text()
s=base.replace('constexpr int width = 4;','constexpr int width = 2;')
a=s.index('        std::fprintf(stderr, "N104 FIXTURE ');b=s.index('    }\n};',a)
s=s[:a]+'''        for(int row=0;row<width;row++) {
            std::fprintf(stderr,"Q2_COHORT FIXTURE_IDS %s row=%d ids=",name,row);
            for(int k=0;k<ranks;k++) std::fprintf(stderr,"%s%d",k?",":"",ids[row*ranks+k]);
            std::fprintf(stderr,"\\n");
        }
        std::fprintf(stderr,"Q2_COHORT FIXTURE %s overlap=%d min_margin=%.8g\\n",name,pair_overlap[0],shortlist_margin);
'''+s[b:]
s=s.replace('generic, split, router','generic, router')
a=s.index('        } else if (mode == Mode::split) {');b=s.index('        } else {',a)
s=s[:a]+s[b:]
s=s.replace('(mode == Mode::split ? 1 : width)','width').replace('mode == Mode::generic ? 3 : 3*width','3').replace('n104_generic_q4','cohort_native_n2split')
a=s.index('using LaunchCounter =');b=s.index('} // namespace',a)
s=s[:a]+Path('probe.inc').read_text()+s[b:]
s=s.replace(', split(weights, Mode::split)','').replace('        split.allocate(cuda);\n','').replace('compare(cuda, generic, split, router, f)','compare(cuda, generic, router, f)').replace('perf(cuda, generic, split, f)','perf(cuda, generic, f)')
a=s.index('    require(prior_q3_count()');b=s.index('    ggml_backend_free',a)
s=s[:a]+'''    const auto done=snapshot();
    std::fprintf(stderr,"Q2_COHORT FINISH row0=%llu row1=%llu pair=%llu skip=%llu\\n",done[0],done[1],done[2],done[3]);
'''+s[b:]
s=s.replace('    ggml_backend_t cuda = nullptr;', '''    if(!offline) {
        require(setenv("GGML_CUDA_OCCAMY_MOE_Q2_COUNT","1",1)==0,"enable physical V3 counts");
        require(getenv("GGML_CUDA_OCCAMY_N2SPLIT") && std::strcmp(getenv("GGML_CUDA_OCCAMY_N2SPLIT"),"1")==0,"original N2SPLIT disabled");
    }
    ggml_backend_t cuda = nullptr;''')
Path('operator.cpp').write_text(s)
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
