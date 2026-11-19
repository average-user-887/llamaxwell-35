from pathlib import Path
import difflib
base=Path('operator.base.cpp').read_text()
s=base.replace('Count count=nullptr;','Count count=nullptr;\nusing Down4Count=unsigned long long (*)();\nDown4Count down4_count=nullptr;')
s=s.replace('std::array<unsigned long long,4>','std::array<unsigned long long,5>').replace('count(2),count(3)}','count(2),count(3),down4_count()}')
a=s.index('    for(size_t i=0;i<before.size();')
b=s.index('\n}',a)
s=s[:a]+'''    const unsigned n=unsigned(runs);
    const Counters expected={0,0,n,n,on?n:0u};
    for (size_t i=0;i<before.size();++i)
        require(after[i]-before[i]==expected[i],"physical row, pair, acknowledgment or four-row down dispatch mismatch");'''+s[b:]
s=s.replace('    Dl_info info{};','''    down4_count=reinterpret_cast<Down4Count>(dlsym(h,"ggml_cuda_occamy_q2_down4_count"));
    Dl_info info{};
    require(down4_count && dladdr(reinterpret_cast<void*>(down4_count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected four-row down counter provider");''')
s=s.replace('"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS",on?"1":"0"','"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS","1"').replace('            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"','            setenv("GGML_CUDA_OCCAMY_MOE_Q2_DOWN4",on?"1":"0",1)==0 &&\n            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"')
s=s.replace('set_mode(on,on);','set_mode(on,true);').replace('        if(on) {\n            Ids','        {\n            Ids')
s=s.replace('Q2_COHORT ROUTES %s row=%d ids=",f.name,r','Q2_PAIR ROUTES %s mode=%s row=%d ids=",f.name,on?"down4":"cohorts",r')
s=s.replace('Q2_COHORT TIME','Q2_PAIR TIME').replace('Q2_COHORT FINISH','Q2_PAIR FINISH').replace('skip=%llu us','skip=%llu down4=%llu us').replace('skip=%llu\\n','skip=%llu down4=%llu\\n').replace('on?"cohorts":"n2split"','on?"down4":"cohorts"').replace('after[3]-before[3],us','after[3]-before[3],after[4]-before[4],us').replace('done[2],done[3]);','done[2],done[3],done[4]);')
s = s.replace('constexpr int reps = 96;', 'constexpr int reps = 32768;')
s = s.replace('batch(false,8);batch(true,8);','batch(false,4096);batch(true,4096);')
s = s.replace('const auto start=std::chrono::steady_clock::now();','const auto epoch0=std::chrono::system_clock::now();\n        const auto start=std::chrono::steady_clock::now();')
s = s.replace('const auto end=std::chrono::steady_clock::now();','const auto end=std::chrono::steady_clock::now();\n        const auto epoch1=std::chrono::system_clock::now();')
i = s.index('        std::fprintf(stderr,"Q2_PAIR TIME')
s = s[:i] + '        std::fprintf(stderr,"Q2_DOWN4 WINDOW %s mode=%s reps=%d begin=%.6f end=%.6f\\n",f.name,on?"down4":"cohorts",n,std::chrono::duration<double>(epoch0.time_since_epoch()).count(),std::chrono::duration<double>(epoch1.time_since_epoch()).count());\n' + s[i:]
Path('operator.cpp').write_text(s)
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
