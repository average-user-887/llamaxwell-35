from pathlib import Path
import difflib
base=Path('operator.base.cpp').read_text()
s=base.replace('Count count=nullptr;','Count count=nullptr;\nusing GateUpCount=unsigned long long (*)();\nGateUpCount gate_up_count=nullptr;')
s=s.replace('std::array<unsigned long long,4>','std::array<unsigned long long,5>').replace('count(2),count(3)}','count(2),count(3),gate_up_count()}')
a=s.index('    for(size_t i=0;i<before.size();')
b=s.index('\n}',a)
s=s[:a]+'''    const unsigned n=unsigned(runs);
    const Counters expected={0,0,n,n,on?n:0u};
    for (size_t i=0;i<before.size();++i)
        require(after[i]-before[i]==expected[i],"physical row, pair, acknowledgment or gate/up dispatch mismatch");'''+s[b:]
s=s.replace('    Dl_info info{};','''    gate_up_count=reinterpret_cast<GateUpCount>(dlsym(h,"ggml_cuda_occamy_q2_gate_up_count"));
    Dl_info info{};
    require(gate_up_count && dladdr(reinterpret_cast<void*>(gate_up_count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected gate/up counter provider");''')
s=s.replace('"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS",on?"1":"0"','"GGML_CUDA_OCCAMY_MOE_Q2_COHORTS","1"').replace('            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"','            setenv("GGML_CUDA_OCCAMY_MOE_Q2_GATE_UP",on?"1":"0",1)==0 &&\n            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS"')
s=s.replace('set_mode(on,on);','set_mode(on,true);').replace('        if(on) {\n            Ids','        {\n            Ids')
s=s.replace('Q2_COHORT ROUTES %s row=%d ids=",f.name,r','Q2_PAIR ROUTES %s mode=%s row=%d ids=",f.name,on?"gateup":"cohorts",r')
s=s.replace('Q2_COHORT TIME','Q2_PAIR TIME').replace('Q2_COHORT FINISH','Q2_PAIR FINISH').replace('skip=%llu us','skip=%llu gateup=%llu us').replace('skip=%llu\\n','skip=%llu gateup=%llu\\n').replace('on?"cohorts":"n2split"','on?"gateup":"cohorts"').replace('after[3]-before[3],us','after[3]-before[3],after[4]-before[4],us').replace('done[2],done[3]);','done[2],done[3],done[4]);')
Path('operator.cpp').write_text(s)
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
