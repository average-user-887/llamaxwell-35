import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
source = here.parent / 'night-gdn-t2-q8-fast/occamy-gdn.cu'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'cdab6a823ca92a648f83dadf35379151347bdafac564949d0c759d36827728e9'
original = source.read_text()
def once(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b, 1)

start = original.index('template <int T2>\n__global__ void __launch_bounds__(NT, 3) gdn_fused2')
end = original.index('\nstruct DevState {', start)
old = original[start:end]
diagnostic = once(old, 'template <int T2>\n__global__ void __launch_bounds__(NT, 3) gdn_fused2(const Params2 PP) {', '''__global__ void __launch_bounds__(NT, 3) gdn_fused2_clock(const Params2 PP, volatile unsigned * marks) {
    constexpr int T2 = 2;
    n70_mark<0>(marks);''')
before, after = diagnostic.split('    grid_barrier(s0);', 1)
after = once(after, '    grid_barrier(s0);', '    n70_mark<3>(marks);\n    grid_barrier(s0);\n    n70_mark<4>(marks);')
diagnostic = before + '    n70_mark<1>(marks);\n    grid_barrier(s0);\n    n70_mark<2>(marks);' + after
assert diagnostic.endswith('}\n')
diagnostic = diagnostic[:-2] + '''    n70_mark<5>(marks);

}
'''
mark = '''// Read this warp's SM-local counter. No clocks from different CTAs are subtracted.
template <int mark>
static __device__ __forceinline__ void n70_mark(volatile unsigned * samples) {
    unsigned tid, cta;
    asm volatile("mov.u32 %0, %%tid.x; mov.u32 %1, %%ctaid.x;" : "=r"(tid), "=r"(cta));
    if ((tid & 31) == 0) {
        unsigned ticks;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(ticks) :: "memory");
        samples[(cta * NW + (tid >> 5)) * 6 + mark] = ticks;
    }
}
'''
out = original[:end] + '\n' + mark + diagnostic + original[end:]
anchor = '// GDN2: MTP1 verify window (one sequence, two tokens) as one gdn_fused2<1> launch.'
out = once(out, anchor, (here/'host.inc').read_text() + '\n' + anchor)
anchor = '    if (d.q8a) {'
out = once(out, anchor, '''    static const char * const clock_mode = std::getenv("GGML_CUDA_OCCAMY_Q8_PHASE_CLOCK");
    if (clock_mode) {
        GGML_ASSERT(d.q8a && !events);
        n70_launch(clock_mode, ctx.device, d.gridT2, st, PP, m.wqkv->name);
    } else if (d.q8a) {''')
(here/'occamy-gdn.cu').write_text(out)
