#!/usr/bin/env python3
"""gdb sampling, not perf. usage: e9_aggregate.py <sampledir>
Per sample: main thread = 'Thread 1'. Decode-phase sample = main thread stack contains server_context_impl::decode (llama_decode path).
Inclusive presence = share of decode-phase samples whose main-thread stack contains the function. Other threads reported separately (only those with ggml/llama frames)."""
import sys,re,glob,collections
files=sorted(glob.glob(sys.argv[1]+'/s*.txt')); incl=collections.Counter(); inner=collections.Counter(); phase=collections.Counter(); other=collections.Counter(); nd=0
JUNK={'??','free_modules_db','derivation_compare','_start','seekdir','__libc_start_main','__libc_start_call_main'}
def named(fr): return [f for f in fr if f not in JUNK]
for f in files:
    txt=open(f).read(); thr=re.split(r'\n(?=Thread \d+ )',txt); main=None
    for t in thr:
        m=re.match(r'Thread (\d+) ',t)
        if not m: continue
        fr=re.findall(r'^#\d+\s+(?:0x[0-9a-f]+ in )?(\S+)',t,re.M)
        if m.group(1)=='1': main=fr
        else:
            for x in set(named(fr)):
                if x.startswith(('ggml','llama','cuda','cu')): other[x]+=1
    if main is None: phase['no main']+=1; continue
    if any('server_context_impl::decode' in x or 'llama_decode' in x for x in main):
        phase['decode']+=1; nd+=1; incl.update(set(named(main)))
        # innermost frame, classified
        top=main[0]; nm=named(main); cat=('cudaStreamSynchronize wait' if 'cudaStreamSynchronize' in main or 'cuStreamSynchronize' in main else
          'graph build (llama_context::process_ubatch/graph_build)' if any('graph_build' in x or 'build_graph' in x or 'llm_build' in x for x in main) else
          'ggml_backend_sched_graph_compute/launch' if any('graph_compute' in x for x in main) else
          'sampling' if any('sampl' in x.lower() for x in main) else 'other decode-path: '+(nm[0] if nm else '??'))
        inner[cat]+=1
    else:
        phase['non-decode: '+(next((x for x in main if x not in JUNK),'??'))]+=1
print('samples',len(files)); print('phase of main thread:',dict(phase))
print('TOP inclusive presence on main thread, decode-phase samples (n=%d):'%nd)
for k,v in incl.most_common(18): print('  %5.1f%%  %s'%(100*v/max(nd,1),k))
print('Classification of decode-phase samples:')
for k,v in inner.most_common(): print('  %5.1f%%  %s'%(100*v/max(nd,1),k))
print('Other threads, ggml/llama/cuda frames, share of all samples:')
for k,v in other.most_common(8): print('  %5.1f%%  %s'%(100*v/len(files),k))
