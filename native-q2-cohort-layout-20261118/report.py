"""Verify the original two-row graph, bitwise outputs, co-residency and mirrored physical timings."""
import itertools,json,math,re,sys
from pathlib import Path
from placement import read_maps
P=Path(__file__).resolve().parent
MAPS=['libggml.so.0.26.0','libggml-base.so.0.26.0','libggml-cuda.so']
CASES={'independent':0,'partial':5}
F=r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
def need(ok,why):
    if not ok:raise ValueError(why)
def load(directory,stage):
    raw=(directory/(stage+'.log')).read_text()
    expected={}
    for file in ('RUNTIME.sha256','FROZEN.sha256'):
        for line in (directory/file).read_text().splitlines():
            h,name=line.split('  ',1);short=Path(name).name
            if short in [*MAPS,'operator']:expected[short]=h
    actual={Path(path).name:h for h,path in re.findall(r'^([0-9a-f]{64})  (/(?:runtime/\S+|operator))$',raw,re.M)}
    need(actual==expected and len(actual)==4,'actual runtime or program hash mismatch')
    return raw
def common(raw,old,new):
    need(not re.search(r'^N104 ERROR|CUDA error|GGML_ASSERT|CUDA out of memory|illegal memory access',raw,re.I|re.M),'operator or CUDA error')
    need(sorted(re.findall(r'^N104 MAP /runtime/(\S+)$',raw,re.M))==sorted(MAPS),'runtime map inventory')
    need(raw.splitlines().count('Q2_COHORT MAP /operator /runtime/libggml-cuda.so')==1,'actual counter provider')
    for line in (P/'fixture.txt').read_text().splitlines():need(raw.splitlines().count(line)==1,'seeded fixture or reference routes')
    matches=re.findall(r'^Q2_COHORT residency device=0 sm=24 active=(\d+) regs=63 static=96 dynamic=39424 local=0 ready=1$',raw,re.M)
    need(len(matches)==1 and int(matches[0])>=2,'candidate occupancy or register allocation')
    tested=re.findall(r'^Q2_LAYOUT residency kind=(\d+) device=0 sm=24 active=(\d+) regs=(\d+) static=96 dynamic=39424 local=0 ready=1$',raw,re.M)
    kinds={0:64,1:63,2:64} if new!=400 else {0:64}
    need(len(tested)==len(kinds) and {int(k):int(r) for k,_,r in tested}==kinds and all(int(n)>=2 for _,n,_ in tested),'kernel-specific residency')
    need(raw.splitlines().count(f'Q2_PAIR FINISH row0=0 row1=0 pair={old+new} skip={old+new} layout={new}')==1,'actual physical totals')
    return {'sms':24,'active_per_sm':int(matches[0]),'grid_ctas':48,'ctas_per_cohort':24,'threads_per_cta':512,'registers':63,'control_static_shared_bytes':96,'candidate_static_shared_bytes':96,'candidate_registers':64,'dynamic_shared_bytes':39424,'local_bytes':0}
def numeric(raw):
    hw=common(raw,6,8)
    out={}
    for name,overlap in CASES.items():
        ids=lambda tag:[(int(r),[int(i) for i in line.split(',')]) for r,line in re.findall(rf'^Q2_COHORT {tag} {name} row=(\d+) ids=([0-9,]+)$',raw,re.M)]
        fixed=ids('FIXTURE_IDS')
        ordered=[(mode,int(r),list(map(int,line.split(',')))) for mode,r,line in re.findall(rf'^Q2_PAIR ROUTES {name} mode=(cohorts|layout) row=(\d+) ids=([0-9,]+)$',raw,re.M)]
        need(ordered==[(mode,r,values) for mode in ('cohorts','layout','layout') for r,values in fixed],'actual paired routes differ')
        actual=[(r,v) for mode,r,v in ordered if mode=='layout']
        need([r for r,_ in fixed]==[0,1] and actual==fixed+fixed,'physical route or repeat differs')
        need(all(len(s)==len(set(s))==8 and min(s)>=0 and max(s)<256 for _,s in fixed) and len(set(fixed[0][1])&set(fixed[1][1]))==overlap,'route cardinality or coverage')
        rows=[{'row':int(r),'finite':int(f),'different_words':int(m),'repeat_different_words':int(rm),'max_abs':float(pk)} for r,f,m,rm,pk in re.findall(
            rf'^Q2_COHORT ROW {name} row=(\d+) finite=(\d+) different=(\d+) repeat_different=(\d+) max_abs=({F})$',raw,re.M)]
        need([v['row'] for v in rows]==[0,1] and all(v['finite']==2048 and v['different_words']==v['repeat_different_words']==v['max_abs']==0 for v in rows),'nonfinite or nonidentical outputs')
        out[name]={'ordered_experts':[x[1] for x in fixed],'expert_overlap':overlap,'candidate_router_repeats':2,'output_rows':rows,
                   'stock_reference':'unchanged 369c6b05 two-row cohort instructions on the same native Q2 graph'}
    return {'residency':hw,'results':out,'diagnostic_layout_maps':read_maps(raw),'placement_scope':'Placement recorded by diagnostic kernel variants; physical SM allocation of the untapped timed kernel is not directly observed.'}
def timing(raw):
    hw=common(raw,400,400)
    need('MOE_V3 CHECK' not in raw and 'Q2_PAIR ROUTES ' not in raw and 'Q2_LAYOUT_MAP ' not in raw,'diagnostic work active during performance')
    out={}
    for name in CASES:
        rx=rf'^Q2_PAIR TIME {name} mode=(cohorts|layout) reps=(\d+) row0=(\d+) row1=(\d+) pair=(\d+) skip=(\d+) layout=(\d+) us=({F})$'
        vals=[(m,int(n),int(a),int(b),int(p),int(s),int(g),float(t)) for m,n,a,b,p,s,g,t in re.findall(rx,raw,re.M)]
        a,b=('cohorts','layout') if name=='independent' else ('layout','cohorts')
        need([r[0] for r in vals]==['cohorts','layout',a,b,b,a] and [r[1] for r in vals]==[8,8,96,96,96,96],'timing order or count')
        for mode,n,x,y,p,s,g,t in vals:need((x,y,p,s,g)==(0,0,n,n,n if mode=='layout' else 0) and math.isfinite(t) and t>0,'timed launches or duration')
        pairs=[]
        for pair in (vals[2:4],vals[4:6]):
            stage={row[0]:row[-1] for row in pair};pairs.append(stage['cohorts']-stage['layout'])
        out[name]={'interval_columns':['mode','reps','old_row0','old_row1','combined_launches','second_callbacks_skipped','layout_launches','mean_us'],'all_intervals':vals,'cohorts_minus_layout_us':pairs,
                   'projected_ms_across_40_layers':[v*.04 for v in pairs],'materiality_pass':min(pairs)>=18}
    return {'residency':hw,'results':out}
def selftest():
    raw=(P/'fixture.txt').read_text()+''.join(f'N104 MAP /runtime/{m}\n' for m in MAPS)
    raw+='Q2_COHORT MAP /operator /runtime/libggml-cuda.so\nQ2_COHORT residency device=0 sm=24 active=2 regs=63 static=96 dynamic=39424 local=0 ready=1\n'
    raw+='Q2_LAYOUT residency kind=0 device=0 sm=24 active=2 regs=64 static=96 dynamic=39424 local=0 ready=1\n'
    diag=raw;perf=raw
    for kind,regs in [(1,63),(2,64)]:
        diag+=f'Q2_LAYOUT residency kind={kind} device=0 sm=24 active=2 regs={regs} static=96 dynamic=39424 local=0 ready=1\n'
    for fixture in CASES:
        for sample in range(2):
            for major in range(2):
                for physical in range(48):
                    y,x=divmod(physical,24)
                    local,row=divmod(physical,2)
                    if major: row,local=y,x
                    diag+=f'Q2_LAYOUT_MAP fixture={fixture} sample={sample} major={major} x={x} y={y} row={row} local={local} sm={x}\n'
    for name in CASES:
        rows=[l for l in raw.splitlines() if l.startswith('Q2_COHORT FIXTURE_IDS '+name+' ')]
        diag+=''.join(x.replace('Q2_COHORT FIXTURE_IDS','Q2_PAIR ROUTES').replace(' row=',f' mode={mode} row=')+'\n' for mode in ('cohorts','layout','layout') for x in rows)
        diag+=''.join(f'Q2_COHORT ROW {name} row={i} finite=2048 different=0 repeat_different=0 max_abs=0\n' for i in range(2))
        a,b=('cohorts','layout') if name=='independent' else ('layout','cohorts')
        for mode,n in zip(['cohorts','layout',a,b,b,a],[8,8,96,96,96,96]):
            old=n if mode=='cohorts' else 0;new=n-old
            perf+=f'Q2_PAIR TIME {name} mode={mode} reps={n} row0=0 row1=0 pair={n} skip={n} layout={new} us={165 if old else 138}\n'
    diag+='Q2_PAIR FINISH row0=0 row1=0 pair=14 skip=14 layout=8\n'
    perf+='Q2_PAIR FINISH row0=0 row1=0 pair=800 skip=800 layout=400\n'
    numeric(diag);need(all(v['materiality_pass'] for v in timing(perf)['results'].values()),'positive synthetic gate')
    need(not timing(perf.replace('us=138','us=148'))['results']['independent']['materiality_pass'],'synthetic speed miss')
    for fn,bad in [(numeric,diag.replace('local=0 sm=0','local=1 sm=0',1)),(numeric,diag.replace('different=0','different=1',1)),(numeric,diag.replace('active=2','active=1')),(timing,perf.replace('skip=96','skip=95',1))]:
        try:fn(bad)
        except ValueError:continue
        raise ValueError('incorrect data passed')
    print('CPU-only graph, route, exact row, occupancy and mirrored performance parser checks passed.')
def export(d):
    status=dict((k,int(v)) for k,v in (line.split('=',1) for line in (d/'status.txt').read_text().splitlines()))
    out={'status':status,'bounded_wall_s':status['finished_epoch']-status['started_epoch'],'build':json.loads((P/'build.json').read_text()),'materiality_us':18,'scope':'Synchronized complete graph; original paired cohort vs row-major block schedule in a two-row cohort kernel. Forty-layer sums are projections.'}
    for stage,fn in [('check',numeric),('perf',timing)]:
        p=d/(stage+'.log')
        if not p.exists():out[stage]={'executed':False};continue
        raw=p.read_text(errors='replace')
        v={'executed':True,'observed_lines':[s for s in raw.splitlines() if s.startswith(('N104 ','Q2_COHORT ','Q2_PAIR ','Q2_LAYOUT ','Q2_LAYOUT_MAP ')) and len(s)<480]}
        try:v['validated']=fn(load(d,stage))
        except (ValueError,OSError) as e:v['refusal']=str(e) if isinstance(e,ValueError) else type(e).__name__
        out[stage]=v
    out['complete_measurement']=json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else None
    (d/'portable-receipt.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    command,*args=sys.argv[1:]
    if command=='--check':selftest()
    elif command=='--numeric':print(json.dumps(numeric(load(Path(args[0]),'check')),indent=2))
    elif command=='--export':export(Path(args[0]))
    elif command=='--report':
        d=Path(args[0]);n=numeric(load(d,'check'));t=timing(load(d,'perf'))
        need(n['residency']==t['residency'],'device residency differs')
        rec={'technical_pass':True,'materiality_pass':all(v['materiality_pass'] for v in t['results'].values()),'threshold_us_per_layer':18,
             'numeric':n,'timing':t}
        text=json.dumps(rec,indent=2,allow_nan=False)+'\n'
        if (d/'result.json').exists():need(json.loads((d/'result.json').read_text())==json.loads(text),'saved result differs')
        else:(d/'result.json').write_text(text)
        print(text)
    else:raise SystemExit('unexpected command')
