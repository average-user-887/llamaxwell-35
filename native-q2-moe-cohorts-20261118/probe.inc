using Count = unsigned long long (*)(unsigned);
using ReadIds = void (*)(int *);
Count count=nullptr;
ReadIds read_ids=nullptr;
using Counters=std::array<unsigned long long,4>;
Counters snapshot() {return {count(0),count(1),count(2),count(3)};}
void checked_delta(Counters before, Counters after, int runs, bool on) {
    for(size_t i=0;i<before.size();++i) require(after[i]-before[i]==((i>=2)==on?unsigned(runs):0u),"physical path or second callback mismatch");
}
void load_counter() {
    void * h=dlopen("/runtime/libggml-cuda.so",RTLD_NOLOAD|RTLD_NOW);
    require(h,"candidate DSO not mapped");
    count=reinterpret_cast<Count>(dlsym(h,"ggml_cuda_occamy_q2_counter"));
    read_ids=reinterpret_cast<ReadIds>(dlsym(h,"ggml_cuda_occamy_q2_router"));
    Dl_info info{};
    require(count && read_ids && dladdr(reinterpret_cast<void*>(count),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected counter provider");
    char exe[256]{};
    require(readlink("/proc/self/exe",exe,sizeof(exe)-1)>0 && std::strcmp(exe,"/operator")==0,"unexpected operator executable");
    require(snapshot()==Counters{},"new process counters dirty");
    std::fprintf(stderr,"Q2_COHORT MAP /operator /runtime/libggml-cuda.so\n");
    dlclose(h);
}
void set_mode(bool on,bool diag=false) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q2_COHORTS",on?"1":"0",1)==0 &&
            setenv("GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS",diag?"1":"0",1)==0,"cannot select typed mode");
}
void compare(ggml_backend_t cuda,Graph & graph,Graph & router,const Fixture & f) {
    set_mode(false);
    router.use(f);
    router.run(cuda);
    Ids generic_ids{};
    for(int r=0;r<width;r++) ggml_backend_tensor_get(router.output,generic_ids.data()+r*ranks,r*router.output->nb[1],ranks*sizeof(int32_t));
    require(generic_ids==f.ids,"generic ordered expert choices differ from CPU");
    Output base{},trial{},repeat{};
    auto run=[&](bool on,Output & dst){
        set_mode(on,on);graph.use(f);
        const auto before=snapshot();graph.run(cuda);checked_delta(before,snapshot(),1,on);
        ggml_backend_tensor_get(graph.output,dst.data(),0,sizeof(dst));
        if(on) {
            Ids ids{};read_ids(ids.data());require(ids==f.ids,"physical Q2 candidate expert choices differ from CPU");
            for(int r=0;r<width;r++) {
                std::fprintf(stderr,"Q2_COHORT ROUTES %s row=%d ids=",f.name,r);
                for(int k=0;k<ranks;k++) std::fprintf(stderr,"%s%d",k?",":"",ids[r*ranks+k]);
                std::fprintf(stderr,"\n");
            }
        }
    };
    run(false,base);run(true,trial);run(true,repeat);
    for(int row=0;row<width;row++) {
        int different=0,repeat_diff=0,finite=0;double peak=0.;
        for(int i=row*hidden;i<(row+1)*hidden;i++) {
            finite+=std::isfinite(base[i]) && std::isfinite(trial[i]) && std::isfinite(repeat[i]);
            different+=std::memcmp(&base[i],&trial[i],sizeof(float))!=0;
            repeat_diff+=std::memcmp(&trial[i],&repeat[i],sizeof(float))!=0;
            peak=std::max(peak,std::abs(double(base[i])-trial[i]));
        }
        std::fprintf(stderr,"Q2_COHORT ROW %s row=%d finite=%d different=%d repeat_different=%d max_abs=%.10g\n",
                     f.name,row,finite,different,repeat_diff,peak);
        require(finite==hidden && different==0 && repeat_diff==0,"nonfinite or nonidentical row");
    }
    set_mode(false);
}
void perf(ggml_backend_t cuda,Graph & graph,const Fixture & f) {
    auto batch=[&](bool on,int n) {
        set_mode(on);graph.use(f);const auto before=snapshot();
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<n;i++) graph.run(cuda);
        const auto end=std::chrono::steady_clock::now();
        auto after=snapshot();checked_delta(before,after,n,on);
        Output v{};ggml_backend_tensor_get(graph.input,v.data(),0,sizeof(v));
        require(std::memcmp(v.data(),f.input.data(),sizeof(v))==0,"measured graph mutated input");
        const double us=std::chrono::duration<double,std::micro>(end-start).count()/n;
        std::fprintf(stderr,"Q2_COHORT TIME %s mode=%s reps=%d row0=%llu row1=%llu pair=%llu skip=%llu us=%.6f\n",
                     f.name,on?"cohorts":"n2split",n,after[0]-before[0],after[1]-before[1],after[2]-before[2],after[3]-before[3],us);
    };
    batch(false,8);batch(true,8);
    const bool first=std::strcmp(f.name,"partial")==0;
    batch(first,reps);batch(!first,reps);batch(!first,reps);batch(first,reps);
    set_mode(false);
}
