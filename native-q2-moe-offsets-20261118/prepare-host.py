from pathlib import Path
import difflib
base=Path('operator.base.cpp').read_text()
s=base.replace('Count count=nullptr;','Count count=nullptr;\nusing OffsetsCount=unsigned long long (*)();\nOffsetsCount offsets_count=nullptr;')
s=s.replace('std::array<unsigned long long,4>','std::array<unsigned long long,5>').replace('count(2),count(3)}','count(2),count(3),offsets_count()}')
a=s.index('    for(size_t i=0;i<before.size();')
b=s.index('\n}',a)
s=s[:a]+'''    const unsigned n=unsigned(runs);
    const Counters expected={0,0,n,n,on?n:0u};
    for (size_t i=0;i<before.size();++i)
        require(after[i]-before[i]==expected[i],"physical row, pair, acknowledgment or byte-offset dispatch mismatch");'''+s[b:]
s=s.replace('    Dl_info info{};','''    offsets_count=reinterpret_cast<OffsetsCount>(dlsym(h,"ggml_cuda_occamy_q2_offsets_count"));
    Dl_info info{};
    require(offsets_count && dladdr(reinterpret_cast<void*>(offsets_count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected byte-offset counter provider");''')
s=s.replace('"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS",on?"1":"0"','"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS","1"').replace('            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"','            setenv("GGML_CUDA_OCCAMY_MOE_Q2_OFFSETS",on?"1":"0",1)==0 &&\n            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"')
s=s.replace('set_mode(on,on);','set_mode(on,true);').replace('        if(on) {\n            Ids','        {\n            Ids')
s=s.replace('Q2_COHORT ROUTES %s row=%d ids=",f.name,r','Q2_PAIR ROUTES %s mode=%s row=%d ids=",f.name,on?"offsets":"cohorts",r')
s=s.replace('Q2_COHORT TIME','Q2_PAIR TIME').replace('Q2_COHORT FINISH','Q2_PAIR FINISH').replace('skip=%llu us','skip=%llu offsets=%llu us').replace('skip=%llu\\n','skip=%llu offsets=%llu\\n').replace('on?"cohorts":"n2split"','on?"offsets":"cohorts"').replace('after[3]-before[3],us','after[3]-before[3],after[4]-before[4],us').replace('done[2],done[3]);','done[2],done[3],done[4]);')
Path('operator.cpp').write_text(s)
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
