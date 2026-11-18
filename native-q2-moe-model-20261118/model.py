import importlib.util,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; C=HERE.parent
spec=importlib.util.spec_from_file_location('q8serial',C/'resume-q8-on-serial/serial_probe.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
n88,n9,n18=base.n88,base.n9,base.n18
REF=base.REF; read=n9.read_json
DSO='19607daf02bc6dcd722a0695787046db9b4c6b8e010545598d3ad60649d68d00'
SUP=HERE/'supervise'
def need(ok,why):
    if not ok:raise ValueError(why)
def settings(stage):
    need(stage in ('native','serial'),'unknown stage')
    env,argv=n88.arguments('on') if stage=='native' else base.settings()
    return [*env,'-e','GGML_CUDA_OCCAMY_MOE_Q2_COHORTS=1','-e','GGML_CUDA_OCCAMY_MOE_Q2_COUNT=1','-e','GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS=0'],argv
def check():
    base.check()
    need(n9.hash_file(C/'resume-q2-moe-cohorts/libggml-cuda.so')==DSO,'candidate identity')
    r,=(C/'resume-q2-moe-cohorts/receipts').glob('*/result.json')
    d=read(r);need(d['technical_pass'] and not d['materiality_pass'],'incorrect prior operator result')
    for f in ('cold','warm'):
        n9.read_json(REF/(f+'.json'))
def ready(stage,out):
    env,args=settings(stage)
    for name,expected in [('environment.args',env),('server.args',args),('child-server.args',['/candidate/llama-server',*args])]:
        need((out/name).read_text().splitlines()==expected,f'{name}: typed vector differs')
    need(sorted((out/'child-selected-environment.txt').read_text().splitlines())==sorted(env[1::2]),'child flags differ, inactive diagnostic or extra profile required')
    pid=int((out/'witness/child.pid').read_text());need(pid>1,'server must be the owned child')
    maps=base.maps(out/'runtime-library-hashes.txt')
    expect=base.maps(REF/'runtime-library-hashes.txt')
    exe=expect.pop('/proc/1/exe')
    expect.update({'/candidate/libggml-cuda.so':DSO,str(base.SERVER_IMPL):n9.hash_file(base.SERVER_IMPL),f'/proc/{pid}/exe':exe})
    need(maps==expect,'actual executable/implementation/library digest or path differs')
    need((out/'supervisor.sha256').read_text().split()==[n9.hash_file(SUP),'/proc/1/exe'],'supervisor PID1 identity')
    start=(out/'server-start.log').read_text()
    need(all(s in start for s in n9.START_MARKERS) and not n88.short.CUDA_ERROR.search(start),'42/42 offload, CUDA, or slot startup differs')
    need(all(set(re.findall(p,start,re.M))=={'8'} for p in n88.n48.WIDTHS),'effective expert count differs')
    shapes={'n_ctx':['84224'],'n_seq_max':['1'],'n_ubatch':['1024'],'n_rs_seq':['0'],'n_outputs_max_per_seq':['1']}
    if stage=='native':
        shapes={k:[*v,*v] for k,v in shapes.items()}
        shapes['n_rs_seq']=['1','0'];shapes['n_outputs_max_per_seq']=['2','1']
    for k,v in shapes.items():
        need(re.findall(rf'llama_context: {k}\s+=\s*(\d+)\b',start)==v,f'actual {k} differs')
    need(('creating MTP draft context' in start)==(stage=='native'),'actual native or serial context differs')
    n9.require_admission(out/'admission.txt')
    # The returning mapper reports the actual server, never the supervisor.
    return start,maps
def counters(stage,out,metrics):
    vals=[read(out/'witness'/(n+'.json')) for n in ['start','cold','warm']]
    pid=int((out/'witness/child.pid').read_text())
    need(all(v['child']==pid and all(isinstance(v[k],int) and v[k]>=0 for k in ('combined','row0','row1','skip')) for v in vals),'counter witness child or type')
    result={}
    for phase,a,b in zip(('cold','warm'),vals,vals[1:]):
        diff={k:b[k]-a[k] for k in ['row0','row1','combined','skip']}
        need(min(diff.values())>=0 and diff['row0']==diff['row1']==0 and diff['combined']==diff['skip'],'unpaired dispatch or legacy fallback during measured request')
        drafts=int(metrics[phase].get('draft_n',0) or 0)
        need(diff['combined']==40*drafts,f'{phase}: physical count differs from forty per actual offered draft; unexplained preparation receives no credit')
        result[phase]={'physical_deltas':diff,'offered_drafts':drafts,'expected_generation_at_40_target_layers':40*drafts,'extra_unassigned_q2_calls':0,'coverage_scope':'aggregate count over complete request; no individually named layer timings or counters'}
    return {'at_ready':{k:v for k,v in vals[0].items() if k!='child'},'raw_boundaries':[{k:x for k,x in v.items() if k!='child'} for v in vals],'phases':result}
def collect(stage,out):
    if stage=='serial':
        original=base.ready;base.ready=lambda v:ready(stage,v)
        try: result=base.collect(out)
        finally:base.ready=original
    else:
        original=n88.startup;n88.startup=lambda _stage,v:ready(stage,v)
        try: result=n88.inspect_perf('on',out)
        finally:n88.startup=original
    metrics={};parity={}
    for phase in ('cold','warm'):
        a=read(out/(phase+'.json')); ref=read(REF/(phase+'.json'));v=result['metrics'][phase];t=a['timings']
        metrics[phase]={k:v[k] for k in ('n_out','decode_toks_s','request_wall_s','ttft_s','decode_s','cache_n','evaluated_prompt_n')}
        metrics[phase].update(draft_n=t.get('draft_n'),draft_n_accepted=t.get('draft_n_accepted'),decode_ms_per_round=t['predicted_ms']/t['draft_n'] if t.get('draft_n') else None,finish_reason=a['finish_reason'],truncated=a['truncated'])
        parity[phase]={'all_ids_equal':a['token_ids']==ref['token_ids'],'full_text_equal':a['text']==ref['text'],'compared_ids':len(a['token_ids']),
                       'first_id_difference_1_based':n9.first_mismatch(a['token_ids'],ref['token_ids'])}
        if stage=='native':
            need(parity[phase]['all_ids_equal'] and parity[phase]['full_text_equal'],'native output differs from archived N88')
            need(t['draft_n']==ref['timings']['draft_n'] and t['draft_n_accepted']==ref['timings']['draft_n_accepted'],'native proposal/accept totals differ from identical-context N88')
    count=counters(stage,out,metrics)
    full=(out/'server.log').read_text()
    sched=re.findall(r'^Q2_COHORT residency device=0 sm=24 active=(\d+) regs=63 static=96 dynamic=39424 local=0 ready=1$',full,re.M)
    need((len(sched)==1 and int(sched[0])>=2) if stage=='native' else (not sched or (len(sched)==1 and int(sched[0])>=2)),'physical Maxwell admission differs')
    return {'mode':stage,'technical_pass':True,'semantic_score':None,'actual_server_software_sha256':{'exe' if k.startswith('/proc/') else Path(k).name:v for k,v in ready(stage,out)[1].items()},
            'metrics':metrics,'phase_counter_evidence':count,'direct_id_readback_enabled':False,'accepted_experts':8,'outputs_vs_archived_q8_native':parity,
            'native_routes':result.get('routes'),'observed_q2_residency':list(map(int,sched)),'serial_sparse_widths':result.get('warm_serial_sparse_widths')}
def finished(p):
    st=(p/'status.txt').read_text().splitlines()
    need({'runner_exit=0','cleanup_exit=0'}<=set(st),'receipt did not finish')
    x=read(p/'result.json');need(collect(x['mode'],p)==x,'result no longer reproduces');return x
def compare(serial,native):
    a,b=finished(serial),finished(native);need(a['mode']=='serial' and b['mode']=='native','misordered pair')
    rows={}
    for ph in ('cold','warm'):
        x,y=a['metrics'][ph],b['metrics'][ph]; z=read(REF/(ph+'.json'))
        u,v=read(serial/(ph+'.json')),read(native/(ph+'.json'))
        old_t=z['timings']; old_ms=old_t['predicted_ms']/old_t['draft_n']
        rows[ph]={'serial':x,'native':y,'native_over_serial_tps':y['decode_toks_s']/x['decode_toks_s'],'serial_minus_native_wall_s':x['request_wall_s']-y['request_wall_s'],
                  'native_vs_n88_host_ms_per_round_saved':old_ms-y['decode_ms_per_round'],'native_over_historical_n88_tps':y['decode_toks_s']/old_t['predicted_per_second'],
                  'native_text_matches_archived_n88':b['outputs_vs_archived_q8_native'][ph]['full_text_equal'],'strict_first_id_difference_1_based':n9.first_mismatch(u['token_ids'],v['token_ids']),'strict_text_equal':u['text']==v['text']}
    return {'same_candidate_cuda_sha256':DSO,'same_cohort_flags':True,'semantic_score':None,'strict_warm_2x':rows['warm']['native_over_serial_tps']>=2,'measured':rows}
def finish(stage,out,rc,cleanup,secs):
    p=out/'sanitized';p.mkdir()
    obs={}
    for phase in ('cold','warm'):
        f=out/(phase+'.json')
        if f.exists():
            d=read(f);obs[phase]={k:d[k] for k in ('n_out','wall_s','ttft_s','finish_reason') if k in d}
    snapshots={f.stem:{k:v for k,v in read(f).items() if k!='child'} for f in (out/'witness').glob('*.json')}
    err=None
    if rc and (out/'run.log').is_file():
        e=re.findall(r'^Q2 model: ([A-Za-z0-9 .,;_=/-]{1,180})$',(out/'run.log').read_text(),re.M)
        if e:err=e[-1]
    n9.write_json(p/'model-receipt.json',{'mode':stage,'runner_exit':rc,'cleanup_exit':cleanup,'bounded_wall_s':secs,'observations':obs,'snapshots':snapshots,'refusal':err,'validated':read(out/'result.json') if not rc and not cleanup else None})
if __name__=='__main__':
    cmd,*args=sys.argv[1:]
    try:
        if cmd=='check':check()
        elif cmd in ('env','args'):print('\n'.join(settings(args[0])[cmd=='args']))
        elif cmd=='ready':ready(args[0],Path(args[1]))
        elif cmd=='verify':n9.write_json(Path(args[1])/'result.json',collect(args[0],Path(args[1])))
        elif cmd=='compare':print(json.dumps(compare(*map(Path,args)),indent=2))
        elif cmd=='finalize':finish(args[0],Path(args[1]),*map(int,args[2:]))
        else:raise ValueError('unknown mode')
    except (ValueError,KeyError,OSError) as e:raise SystemExit('Q2 model: '+str(e)) from None
