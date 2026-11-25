import sys,importlib.util
spec=importlib.util.spec_from_file_location('c','/tmp/qa-prefill-card/scripts/cache_reuse.py')
import re,collections
src=open('/tmp/qa-prefill-card/scripts/cache_reuse.py').read().split('for p in sys.argv')[0]
exec(src)
for p in sys.argv[1:]:
    T,_=parse(p); rows=[t for t in T.values() if 'pe_ms' in t]
    def b(name,f):
        s=[r for r in rows if f(r)]
        print(f"  {name:38s} n={len(s):5d} proc_tok={sum(r['pe_tok'] for r in s):8d} s={sum(r['pe_ms'] for r in s)/1e3:7.1f} tok/s={sum(r['pe_tok'] for r in s)/max(1e-9,sum(r['pe_ms'] for r in s)/1e3):6.0f}")
    print(p.split('/')[-2])
    b('cold first (sim<=0.5)',lambda r:r['pe_tok']>0.5*r['n_prompt'] and not(r['sim'] and r['sim'][0]>0.5))
    b('cold reuse-miss (sim>0.5, full reproc)',lambda r:r['pe_tok']>0.5*r['n_prompt'] and r['sim'] and r['sim'][0]>0.5)
    b('warm partial >256 tok',lambda r:r['pe_tok']<=0.5*r['n_prompt'] and r['pe_tok']>256)
    b('warm append <=256 tok',lambda r:r['pe_tok']<=0.5*r['n_prompt'] and r['pe_tok']<=256)
    big=sorted(rows,key=lambda r:-r['n_prompt'])[:3]; print('  largest prompts',[ (r['n_prompt'],r['pe_tok']) for r in big])
