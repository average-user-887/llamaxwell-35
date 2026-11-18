"""Only copy typed fields and digest identifiers into portable output."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = {'independent': 0, 'partial': 5}
MODES = ('cohorts','weights')
MAPS = ('libggml.so.0.26.0','libggml-base.so.0.26.0','libggml-cuda.so')
MIN_US = {'independent': -5., 'partial': 30.}
F = r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'

def need(ok, reason):
    if not ok: raise ValueError(reason)

def common(raw, diagnostic):
    need(not re.search(r'^N104 ERROR|CUDA error|GGML_ASSERT|out of memory|illegal memory access',raw,re.I|re.M),'operator or device error')
    need(sorted(re.findall(r'^N104 MAP /runtime/(\S+)$',raw,re.M)) == sorted(MAPS), 'mapped ggml libraries')
    need(raw.splitlines().count('Q2_WEIGHTS MAP /operator /runtime/libggml-cuda.so') == 1,'physical counter provider')
    lines=raw.splitlines()
    for line in (HERE/'fixture.txt').read_text().splitlines(): need(lines.count(line)==1,'seeded reference fixture')
    for tag,smem in [('Q2_COHORT',96),('Q2_WEIGHTS',272)]:
        need(lines.count(f'{tag} residency device=0 sm=24 active=2 regs=63 static={smem} dynamic=39424 local=0 ready=1')==1,'residency or attributed path')
    pair,weights=(6,4) if diagnostic else (800,400)
    need(lines.count(f'Q2_WEIGHTS FINISH row0=0 row1=0 pair={pair} skip={pair} weights={weights}')==1,'physical dispatch and callback totals')
    need('MOE_V3 CHECK' not in raw,'legacy rows executed')
    if not diagnostic: need('Q2_WEIGHTS ROUTES ' not in raw,'synchronous diagnostic readback present during timing')

def reference_ids(raw, name):
    pattern=rf'^Q2_WEIGHTS FIXTURE_IDS {name} row=(\d+) ids=([0-9,]+)$'
    matches=[(int(r),[int(x) for x in v.split(',')]) for r,v in re.findall(pattern,raw,re.M)]
    need([r for r,_ in matches]==[0,1],'invalid reference rows')
    ids=[v for _,v in matches]
    need(all(len(v)==len(set(v))==8 and min(v)>=0 and max(v)<256 for v in ids),'invalid top-eight')
    need(len(set(ids[0])&set(ids[1]))==CASES[name],'fixture intersection')
    return ids

def numeric(raw):
    common(raw,True)
    out={}
    for name, overlap in CASES.items():
        ids=reference_ids(raw,name)
        matches=re.findall(rf'^Q2_WEIGHTS ROUTES {name} mode=(\w+) row=(\d+) ids=([0-9,]+)$',raw,re.M)
        actual=[(m,int(r),[int(i) for i in s.split(',')]) for m,r,s in matches]
        expected=[(m,r,seq) for m in ('cohorts','weights','weights') for r,seq in enumerate(ids)]
        need(actual==expected,'actual ordered routes differ')
        pattern=rf'^Q2_WEIGHTS ROW {name} row=(\d+) finite=(\d+) different=(\d+) repeat_different=(\d+) max_abs=({F})$'
        rows=[tuple(map(int,v[:-1]))+(float(v[-1]),) for v in re.findall(pattern,raw,re.M)]
        need(rows==[(0,2048,0,0,0.),(1,2048,0,0,0.)],'finite/exact 4096 outputs including repeat')
        out[name]=dict(ordered_expert_ids=ids, overlap=overlap, finite_output_words=4096,
                       different_output_words=0, repeat_different_output_words=0, physical_route_checks=3)
    return out

def timing(raw):
    common(raw,False)
    out={}
    for name in CASES:
        pattern=rf'^Q2_WEIGHTS TIME {name} mode=(\w+) reps=(\d+) row0=(\d+) row1=(\d+) pair=(\d+) skip=(\d+) weights=(\d+) us=({F})$'
        records=[]
        for m,n,x,y,p,s,w,us in re.findall(pattern,raw,re.M):
            n=int(n);us=float(us)
            need((int(x),int(y),int(p),int(s),int(w))==(0,0,n,n,n if m=='weights' else 0),'interval dispatch counts')
            need(math.isfinite(us) and us>0,'invalid timing')
            records.append(dict(mode=m,reps=n,mean_us=us))
        a,b=MODES if name=='independent' else MODES[::-1]
        need([v['mode'] for v in records]==['cohorts','weights',a,b,b,a] and
             [v['reps'] for v in records]==[8,8,96,96,96,96],'counterbalanced interval order/count')
        pairs=[]
        for i in (2,4):
            v={r['mode']:r['mean_us'] for r in records[i:i+2]}
            pairs.append(v['cohorts']-v['weights'])
        out[name]=dict(intervals=records,cohorts_minus_weights_us=pairs,min_required_saving_us=MIN_US[name],
                       projected_ms_40_layers=[x*.04 for x in pairs],passed=min(pairs)>=MIN_US[name])
    return out

def checked(d,stage,fn):
    raw=(d/(stage+'.log')).read_text()
    expected={}
    for mf in ('FROZEN.sha256','RUNTIME.sha256'):
        for line in (d/mf).read_text().splitlines():
            h,p=line.split('  ',1)
            if Path(p).name in (*MAPS,'operator'): expected[Path(p).name]=h
    actual=re.findall(r'^([0-9a-f]{64})  (/(?:runtime/\S+|operator))$',raw,re.M)
    need(len(actual)==len(expected)==4 and {Path(p).name:h for h,p in actual}==expected,'mounted binaries mismatch')
    return fn(raw)

def result(d):
    n=checked(d,'check',numeric); t=checked(d,'perf',timing)
    return dict(technical_pass=True,materiality_pass=all(v['passed'] for v in t.values()),numeric=n,timing=t)

def export(d):
    build=json.loads((HERE/'build.json').read_text())
    status=dict((k,int(v)) for k,v in (line.split('=',1) for line in (d/'status.txt').read_text().splitlines()))
    out=dict(build=build,status=status,required_savings_us_per_layer=MIN_US,
             manifest_sha256=hashlib.sha256((HERE/'FROZEN.sha256').read_bytes()).hexdigest())
    for stage,fn in [('check',numeric),('perf',timing)]:
        out[stage]={'executed':(d/(stage+'.log')).is_file()}
        if out[stage]['executed']:
            try: out[stage]['validated']=checked(d,stage,fn)
            except (ValueError,OSError) as e: out[stage]['refusal']=str(e) if isinstance(e,ValueError) else type(e).__name__
    if (d/'result.json').exists(): out['complete_measurement']=json.loads((d/'result.json').read_text())
    (d/'portable-receipt.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

def selftest():
    raw=(HERE/'fixture.txt').read_text()+''.join('N104 MAP /runtime/'+m+'\n' for m in MAPS)
    raw+='Q2_WEIGHTS MAP /operator /runtime/libggml-cuda.so\n'
    for tag,smem in [('Q2_COHORT',96),('Q2_WEIGHTS',272)]:
        raw+=f'{tag} residency device=0 sm=24 active=2 regs=63 static={smem} dynamic=39424 local=0 ready=1\n'
    diag=raw;perf=raw
    for name in CASES:
        for m in ['cohorts','weights','weights']:
            for row,ids in enumerate(reference_ids(raw,name)):
                diag+=f'Q2_WEIGHTS ROUTES {name} mode={m} row={row} ids='+','.join(map(str,ids))+'\n'
        for row in (0,1): diag+=f'Q2_WEIGHTS ROW {name} row={row} finite=2048 different=0 repeat_different=0 max_abs=0\n'
        a,b=MODES if name=='independent' else MODES[::-1]
        for m,n in zip(['cohorts','weights',a,b,b,a],[8,8,96,96,96,96]):
            us=200 if m=='cohorts' else (205 if name=='independent' else 170)
            perf+=f'Q2_WEIGHTS TIME {name} mode={m} reps={n} row0=0 row1=0 pair={n} skip={n} weights={n if m=="weights" else 0} us={us}\n'
    diag+='Q2_WEIGHTS FINISH row0=0 row1=0 pair=6 skip=6 weights=4\n'
    perf+='Q2_WEIGHTS FINISH row0=0 row1=0 pair=800 skip=800 weights=400\n'
    numeric(diag)
    need(all(v['passed'] for v in timing(perf).values()),'inclusive speed boundaries')
    need(not timing(perf.replace('us=205','us=206'))['independent']['passed'],'detect disjoint regression')
    need(not timing(perf.replace('us=170','us=171'))['partial']['passed'],'detect partial miss')
    for fn,bad in [(numeric,diag.replace('different=0','different=1',1)),(numeric,diag.replace('active=2','active=1',1)),
                   (timing,perf.replace('skip=96','skip=95',1))]:
        try:fn(bad)
        except ValueError:continue
        raise ValueError('bad fixture accepted')
    print('CPU parser tested at both inclusive thresholds and against wrong output, occupancy and dispatch.')

if __name__=='__main__':
    cmd,*args=sys.argv[1:]
    if cmd=='--check': selftest()
    elif cmd=='--export': export(Path(args[0]))
    elif cmd=='--numeric': print(json.dumps(checked(Path(args[0]),'check',numeric),indent=2))
    elif cmd=='--report':
        d=Path(args[0]); rec=result(d); txt=json.dumps(rec,indent=2,allow_nan=False)+'\n'
        if (d/'result.json').exists(): need(json.loads((d/'result.json').read_text())==rec,'saved measurement mismatch')
        else: (d/'result.json').write_text(txt)
        print(txt,end='')
    else: raise SystemExit('unrecognized mode')
