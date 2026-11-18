from pathlib import Path

base = Path('../night-gdn-t2-state')
source = (base / 'state.cpp').read_text()
anchor = '        first_choice = decode("badpair",{keep,wrong},true).front();'
assert source.count(anchor) == 1
insert = '''        auto counter = cuda_symbol<unsigned long long (*)(unsigned)>("ggml_cuda_occamy_q2_counter");
        unsigned long long start[4];
        for (unsigned i=0; i<4; ++i) start[i] = counter(i);
''' + anchor + '''
        unsigned long long delta[4];
        for (unsigned i=0; i<4; ++i) delta[i] = counter(i) - start[i];
        std::fprintf(stderr, "Q8CLK_COHORT fallback0=%llu fallback1=%llu paired=%llu skipped=%llu\\n",
                     delta[0], delta[1], delta[2], delta[3]);
        require(delta[0]==0 && delta[1]==0 && delta[2]==40 && delta[3]==40,
                "fixed q2 target cohort counters diverged");
'''
Path('state.cpp').write_text(source.replace(anchor, insert))
Path('dispatch.inc').write_bytes((base / 'dispatch.inc').read_bytes())
