from pathlib import Path
p=Path('../resume-q2-moe-cohorts/operator.cpp')
s=p.read_text().replace('Q2_COHORT ','Q2_WEIGHTS ')
s=s.replace('using Counters=std::array<unsigned long long,4>;', 'using Counters=std::array<unsigned long long,5>;')
s=s.replace('count(2),count(3)}','count(2),count(3),count(4)}')
a=s.index('    for(size_t i=0;i<before.size();++i)');b=s.index('\n}',a)
s=s[:a]+'''    Counters expected{0,0,unsigned(runs),unsigned(runs),on ? unsigned(runs) : 0};
    for(size_t i=0;i<before.size();++i)
        require(after[i]-before[i]==expected[i],"physical path or second callback mismatch");'''+s[b:]
s=s.replace('setenv("GGML_CUDA_OCCAMY_MOE_Q2_COHORTS",on?"1":"0",1)', 'setenv("GGML_CUDA_OCCAMY_MOE_Q2_COHORTS","1",1)==0 &&\n            setenv("GGML_CUDA_OCCAMY_MOE_Q2_SHARED_WEIGHTS",on?"1":"0",1)')
s=s.replace('set_mode(on,on);graph.use(f);','set_mode(on,true);graph.use(f);')
s=s.replace('        if(on) {','        {')
s=s.replace('"Q2_WEIGHTS ROUTES %s row=%d ids=",f.name,r','"Q2_WEIGHTS ROUTES %s mode=%s row=%d ids=",f.name,on?"weights":"cohorts",r')
s=s.replace('pair=%llu skip=%llu us=', 'pair=%llu skip=%llu weights=%llu us=')
s=s.replace('on?"cohorts":"n2split"','on?"weights":"cohorts"')
s=s.replace('after[3]-before[3],us','after[3]-before[3],after[4]-before[4],us')
s=s.replace('pair=%llu skip=%llu\\n",done[0],done[1],done[2],done[3]', 'pair=%llu skip=%llu weights=%llu\\n",done[0],done[1],done[2],done[3],done[4]')
s=s.replace('    if(!offline) {','''    require(independent[0]==0 && partial[0]==5,"changed fixture intersection");
    const Ids & p=fixtures[1].ids;
    int permuted_matches=0;
    for(int i=0;i<ranks;++i) for(int j=0;j<ranks;++j)
        if(i!=j && p[i]==p[ranks+j]) ++permuted_matches;
    require(permuted_matches>=4,"partial fixture must include rank permutation");
    if(!offline) {''')
assert s.count('GGML_CUDA_OCCAMY_MOE_Q2_COHORTS')==1 and s.count('GGML_CUDA_OCCAMY_MOE_Q2_SHARED_WEIGHTS')==1
Path('operator.cpp').write_text(s)
