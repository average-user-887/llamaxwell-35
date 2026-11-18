import difflib
from pathlib import Path
base=Path('operator.base.cpp').read_text()
s=base.replace('Offsets','Layout').replace('offsets','layout').replace('OFFSETS','LAYOUT').replace('byte-offset','block layout')
s=s.replace('ReadIds read_ids=nullptr;','ReadIds read_ids=nullptr;\nusing ReadMap = void (*)(float *);\nReadMap read_map=nullptr;')
s=s.replace('    Dl_info info{};', '''    read_map=reinterpret_cast<ReadMap>(dlsym(h,"ggml_cuda_occamy_q2_layout_map"));
    Dl_info info{};
    require(read_map && dladdr(reinterpret_cast<void*>(read_map),&info) && info.dli_fname &&
            std::strcmp(info.dli_fname,"/runtime/libggml-cuda.so")==0,"unexpected map provider");''')
a=s.index('    set_mode(false);\n}',s.index('void compare('))
s=s[:a]+'''    // Replay mapped kernels separately. The arithmetic reference above uses the actual untapped paths.
    require(setenv("GGML_CUDA_OCCAMY_MOE_Q2_LAYOUT_MAP","1",1)==0,"enable placement measurement");
    for (int sample=0;sample<2;++sample) {
        for (bool major : {false,true}) {
            set_mode(major,true);
            graph.use(f);
            const auto before=snapshot();
            graph.run(cuda);
            checked_delta(before,snapshot(),1,major);
            Output mapped{};
            ggml_backend_tensor_get(graph.output,mapped.data(),0,sizeof(mapped));
            require(std::memcmp(mapped.data(),base.data(),sizeof(mapped))==0,"map witness changed output");
            Ids ids{}; read_ids(ids.data());
            require(ids==f.ids,"map witness changed routes");
            std::array<float,2*24*5> records{};
            read_map(records.data());
            for (int row=0;row<2;++row) {
                for (int local=0;local<24;++local) {
                    const float * t=records.data()+(row*24+local)*5;
                    const int physical=major ? row*24+local : 2*local+row;
                    require(t[0]==physical%24 && t[1]==physical/24 && t[2]==row && t[3]==local &&
                            std::isfinite(t[4]) && t[4]>=0 && t[4]<65536 && t[4]==std::floor(t[4]),
                            "invalid or missing placement record");
                    std::fprintf(stderr,"Q2_LAYOUT_MAP fixture=%s sample=%d major=%d x=%d y=%d row=%d local=%d sm=%d\\n",
                                 f.name,sample,int(major),int(t[0]),int(t[1]),row,local,int(t[4]));
                }
            }
        }
    }
    require(unsetenv("GGML_CUDA_OCCAMY_MOE_Q2_LAYOUT_MAP")==0,"disable placement measurement");
'''+s[a:]
Path('operator.cpp').write_text(s)
Path('operator.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/operator.cpp',tofile='b/operator.cpp')))
