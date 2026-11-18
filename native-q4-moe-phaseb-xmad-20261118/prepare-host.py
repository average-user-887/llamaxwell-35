from pathlib import Path
import difflib
s=Path('operator.base.cpp').read_text()
s=s.replace('LaunchCounter prior_q3_count = nullptr;', '''LaunchCounter prior_q3_count = nullptr, xmad_count = nullptr;
using ReadIds = void (*)(int *);
ReadIds xmad_ids = nullptr;

void set_xmad(bool enabled) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_PHASEB_XMAD", enabled ? "1" : "0", 1)==0,"set xmad mode");
}''')
i=s.index('    Dl_info location{};',s.index('void load_counter()'))
s=s[:i]+'''    xmad_count=reinterpret_cast<LaunchCounter>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_count"));
    xmad_ids=reinterpret_cast<ReadIds>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_last_ids"));
    require(xmad_count && xmad_ids && xmad_count()==0,"missing actual physical counter or router reader");
    Dl_info xm_info{};
    require(dladdr(reinterpret_cast<void *>(xmad_count),&xm_info) && xm_info.dli_fname &&
        std::strcmp(xm_info.dli_fname,"/runtime/libggml-cuda.so")==0, "wrong xmad counter provider");
'''+s[i:]
i=s.index('void compare(')
s=s[:i]+'''struct Error {
    double rms, peak;
    int changed;
};
Error error(const Output & ref, const Output & actual, int row) {
    double diff2=0., magnitude=0., peak=0.;
    int changed=0;
    for (int i=row*hidden;i<(row+1)*hidden;++i) {
        require(std::isfinite(ref[i]) && std::isfinite(actual[i]),"nonfinite comparison row");
        const double d=double(actual[i])-ref[i];
        diff2+=d*d;
        magnitude+=double(ref[i])*ref[i];
        peak=std::max(peak,std::abs(d));
        changed+=std::memcmp(&ref[i],&actual[i],sizeof(float))!=0;
    }
    require(magnitude>0.,"empty comparison norm");
    return {std::sqrt(diff2/magnitude),peak,changed};
}
'''+s[i:]
s=s.replace('    const auto initial_count = cohort_count();','    set_xmad(false);\n    const auto initial_count = cohort_count();')
i=s.index('    set_cohorts(false);\n    for (int row',s.index('Q4_COHORT DIAG'))
s=s[:i]+'''    Output xmad{}, repeated{};
    set_xmad(true);
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","1",1)==0,"activate direct route readback");
    auto initial_xmad=xmad_count();
    bool observed=true;
    for (Output * out : {&xmad,&repeated}) {
        generic.use(f);
        generic.run(cuda);
        ggml_backend_tensor_get(generic.output,out->data(),0,sizeof(Output));
        require(xmad_count()==++initial_xmad && cohort_count()==initial_count+1,"candidate or control physical coverage");
        Ids actual{};
        xmad_ids(actual.data());
        observed=observed && (actual==f.ids);
    }
    const bool identical=std::memcmp(xmad.data(),repeated.data(),sizeof(Output))==0;
    std::fprintf(stderr,"Q4_XMAD ROUTE %s matched=%d repeat_equal=%d launches=2\\n",f.name,observed?32:0,int(identical));
    require(observed && identical,"physical xmad route or repeat result differs");
    set_xmad(false);
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","0",1)==0,"clear direct route readback");
'''+s[i:]
i=s.index('        require(magnitude > 0, "degenerate output");')
s=s[:i]+'''        const Error old=error(base,cohorts,row), generic_err=error(base,xmad,row), v3_err=error(cohorts,xmad,row);
        std::fprintf(stderr,"Q4_XMAD ROW %s row=%d finite=2048 old_rms=%.12g old_peak=%.12g generic_rms=%.12g generic_peak=%.12g v3_rms=%.12g v3_peak=%.12g changed_v3=%d\\n",
            f.name,row,old.rms,old.peak,generic_err.rms,generic_err.peak,v3_err.rms,v3_err.peak,v3_err.changed);
'''+s[i:]
a=s.index('void perf('); b=s.index('} // namespace',a)
s=s[:a]+'''void perf(ggml_backend_t cuda, Graph & generic, Graph &, const Fixture & f) {
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q4_XMAD_DIAG","0",1)==0,"disable route readback");
    set_cohorts(true);
    auto batch=[&](bool xmad,int count) {
        set_xmad(xmad);
        generic.use(f);
        const auto old_count=cohort_count(), new_count=xmad_count();
        const auto start=std::chrono::steady_clock::now();
        for (int i=0;i<count;++i) generic.run(cuda);
        const auto finish=std::chrono::steady_clock::now();
        const auto old_calls=cohort_count()-old_count, new_calls=xmad_count()-new_count;
        Output input_after{};
        ggml_backend_tensor_get(generic.input,input_after.data(),0,sizeof(Output));
        require(std::memcmp(input_after.data(),f.input.data(),sizeof(Output))==0,"timing input changed");
        require(old_calls==unsigned(xmad?0:count) && new_calls==unsigned(xmad?count:0),"timing physical coverage");
        const double us=std::chrono::duration<double,std::micro>(finish-start).count()/count;
        std::fprintf(stderr,"Q4_XMAD TIME %s mode=%s reps=%d old=%llu new=%llu us=%.6f\\n",
                     f.name,xmad?"xmad":"cohorts",count,old_calls,new_calls,us);
    };
    batch(false,8); batch(true,8);
    batch(false,reps); batch(true,reps); batch(true,reps); batch(false,reps);
    set_xmad(false);
    set_cohorts(false);
}
'''+s[b:]
s=s.replace('    ggml_backend_free(cuda);','    std::fprintf(stderr,"Q4_XMAD FINISH launches=%llu\\n",xmad_count());\n    ggml_backend_free(cuda);')
s=s.replace('enum class Mode { generic, split, router };','enum class Mode { generic, split, router, phase };')
s=s.replace('if (mode == Mode::router) {','''if (mode == Mode::phase) {
            auto * xn=ggml_mul(ctx,ggml_rms_norm(ctx,input,1e-6f),weights.norm);
            auto * ids=select(ggml_soft_max(ctx,ggml_mul_mat(ctx,weights.router,xn)));
            auto * xx=ggml_reshape_3d(ctx,xn,hidden,1,width);
            auto * ru=ggml_mul_mat_id(ctx,weights.up,xx,ids);
            auto * rg=ggml_mul_mat_id(ctx,weights.gate,xx,ids);
            auto * routed=ggml_reshape_2d(ctx,ggml_swiglu_split(ctx,rg,ru),ranks*middle,width);
            auto * su=ggml_mul_mat(ctx,weights.shared_up,xn);
            auto * sg=ggml_mul_mat(ctx,weights.shared_gate,xn);
            output=ggml_concat(ctx,routed,ggml_swiglu_split(ctx,sg,su),0);
        } else if (mode == Mode::router) {''')
s=s.replace('mode == Mode::router ? ranks * width : hidden * width','mode == Mode::phase ? (ranks+1)*middle*width : mode == Mode::router ? ranks * width : hidden * width')
s=s.replace('mode == Mode::router ? 0 : mode == Mode::generic ? 3 : 3*width','mode == Mode::phase ? 2 : mode == Mode::router ? 0 : mode == Mode::generic ? 3 : 3*width')
s=s.replace('ReadIds xmad_ids = nullptr;', 'ReadIds xmad_ids = nullptr;\nusing ReadPhase=void (*)(float *);\nReadPhase xmad_phase=nullptr;')
s=s.replace('require(xmad_count && xmad_ids && xmad_count()==0', 'xmad_phase=reinterpret_cast<ReadPhase>(dlsym(handle,"ggml_cuda_occamy_q4_xmad_last_phase"));\n    require(xmad_count && xmad_ids && xmad_phase && xmad_count()==0')
s=s.replace('Graph & router, const Fixture & f)', 'Graph & router, Graph & phase, const Fixture & f)')
s=s.replace('router(weights, Mode::router);','router(weights, Mode::router), phase(weights, Mode::phase);')
s=s.replace('router.allocate(cuda);','router.allocate(cuda);\n        if (diagnostic) phase.allocate(cuda);').replace('split, router, f);','split, router, phase, f);')
i=s.index('    const bool identical=',s.index('Output xmad{}, repeated{};'))
s=s[:i]+'''    std::array<float,width*(ranks+1)*middle> observed_phase{},reference_phase{};
    xmad_phase(observed_phase.data());
    set_xmad(false);
    set_cohorts(false);
    phase.use(f);
    phase.run(cuda);
    ggml_backend_tensor_get(phase.output,reference_phase.data(),0,sizeof(reference_phase));
    for(int row=0;row<width;++row) {
        std::fprintf(stderr,"Q4_XMAD EXPERTS %s row=%d ids=",f.name,row);
        Ids selected{};
        xmad_ids(selected.data());
        for(int k=0;k<ranks;++k) std::fprintf(stderr,"%s%d",k?",":"",selected[row*ranks+k]);
        std::fprintf(stderr,"\\n");
        double sq=0.,ref=0.,peak=0.;
        for(int i=row*(ranks+1)*middle;i<(row+1)*(ranks+1)*middle;++i) {
            require(std::isfinite(observed_phase[i]) && std::isfinite(reference_phase[i]),"nonfinite phase B");
            const double d=double(observed_phase[i])-reference_phase[i];
            sq+=d*d;
            ref+=double(reference_phase[i])*reference_phase[i];
            peak=std::max(peak,std::abs(d));
        }
        require(ref>0.,"degenerate phase B");
        std::fprintf(stderr,"Q4_XMAD PHASE %s row=%d finite=4608 generic_rms=%.12g generic_peak=%.12g\\n",
                     f.name,row,std::sqrt(sq/ref),peak);
    }
'''+s[i:]
i=s.index('        require(shortlist_margin',s.index('    void report() const {'))
s=s[:i]+'''        for (int row=0;row<width;++row) {
            std::fprintf(stderr,"N104 FIXTURE_IDS %s row=%d ids=",name,row);
            for(int k=0;k<ranks;++k) std::fprintf(stderr,"%s%d",k?",":"",ids[row*ranks+k]);
            std::fprintf(stderr,"\\n");
        }
'''+s[i:]
Path('operator.cpp').write_text(s)
base=Path('operator.base.cpp').read_text()
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
