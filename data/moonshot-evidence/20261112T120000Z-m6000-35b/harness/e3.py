#!/usr/bin/env python3
# E3 mixed serving driver. Run under: flock <lock> python3 e3.py <plan>   (plan: smoke|blocks|ubx)
import sys,os,json,time,subprocess,threading,traceback,io,signal,urllib.request
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from evlib import *
import metrics as MX
RUNS=E+'/runs.jsonl'; REQS=E+'/requests.jsonl'; SESS=E+'/sessions.jsonl'
SESSION_ID=now_utc()+'-E3'
fails=[0]; unit_ctr=[0]
def unload_35b():
    r=urllib.request.Request('http://<lab-endpoint>/api/models/unload/qwen36-35b-moe',data=b'',method='POST')
    try: print('unload',urllib.request.urlopen(r,timeout=120).status,flush=True)
    except Exception as ex: print('unload err',ex,flush=True)
def telemetry(path):
    f=open(path,'w'); p=subprocess.Popen(['nvidia-smi','-i',GPU,'--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks_event_reasons.active','--format=csv','-l','1'],stdout=f,stderr=subprocess.DEVNULL); return p,f
def run_unit(srv,kind,fx,cycle,block,pos,deep_slot,warm,n_deep,useg,order_index,extra=None):
    """kind: alone|mixed. deep fixture d0 in deep_slot (restored from slot file), mixed adds uncached 16K (segment useg) in other slot."""
    t_utc=now_utc(); rid='%s-d0-%s-c%s-b%s-p%s-%s%s'%(t_utc,srv.arm,cycle,block,pos,kind,'-warm' if warm else '')
    rd=E+'/runs/'+rid; os.makedirs(rd,exist_ok=False)
    other=1-deep_slot; out=io.StringIO(); err=io.StringIO(); rc=0; row={'schema_version':SCHEMA,'session_id':SESSION_ID,'run_id':rid,'evidence_origin':'new_measurement','cycle_id':'c%s'%cycle,'block':block,'position':pos,
      'order_index':order_index,'variant_id':srv.arm,'workload_id':'e3-'+kind,'config_id':'config-%s-ub%d'%(srv.arm,srv.ub),'server_session':srv.sid,'source_sha':None,'source_ref_reported':SRC[srv.arm],'source_sha_missing_reason':'full SHA not recorded by launcher; short ref from launcher comment',
      'binary_sha256':sha_file(LIBS[srv.arm]),'instrumented':False,'warmup':warm,'start_utc':iso(),'deep_slot':deep_slot,'n_deep_requested':n_deep,'contamination':[],'raw_paths':[rid]}
    if buildup_running(): row['contamination'].append('build-up running at start')
    tel=telemetry(rd+'/telemetry.csv'); t_log=iso(); prompt_d=prompt_for(fx,'d0'); recs=[]
    cmd={'kind':kind,'deep_slot':deep_slot,'other_slot':other,'deep_prompt_sha256':hashlib.sha256(json.dumps(prompt_d).encode()).hexdigest(),'deep_prompt_tokens':len(prompt_d),'n_predict_deep':n_deep,
         'server_argv':srv.cmd,'sampling':{'temperature':0,'ignore_eos':True,'cache_prompt':True,'stream':True}}
    try:
        st,j=http_json('/slots/%d?action=restore'%deep_slot,{'filename':'d0.bin'}); assert st==200,('restore',st,j); cmd['restore']=j; resident0=j.get('n_restored')
        st,j=http_json('/slots/%d?action=erase'%other,{}); cmd['erase_other']=[st,j]
        time.sleep(1.0)
        deep=Stream('deep',prompt_d,deep_slot,n_deep); streams={'deep':deep}; t0=time.monotonic_ns(); arrival={}
        sm=None
        if kind=='mixed':
            p16=prompt_for(fx,'u%02d'%useg); cmd['uncached_segment']='u%02d'%useg; cmd['u_prompt_sha256']=hashlib.sha256(json.dumps(p16).encode()).hexdigest(); cmd['u_prompt_tokens']=len(p16)
            def gate():
                while len(deep.ts)<8 and deep.is_alive(): time.sleep(0.02)
                arrival['t']=time.monotonic_ns()
            sm=Stream('u16k',p16,other,32,start_gate=gate); streams['u16k']=sm
        deep.start(); 
        if sm: sm.start()
        deep.join(); 
        if sm: sm.join()
        t1=time.monotonic_ns()
        M={n:stream_metrics(s,t0) for n,s in streams.items()}
        if any(m['err'] for m in M.values()): raise RuntimeError('stream error '+str({n:m['err'] for n,m in M.items()}))
        json.dump({'t0_monotonic_ns':t0,'arrival_ns':(arrival['t']-t0) if arrival else None,'streams':M},open(rd+'/events.json','w'))
        res={'wall_ns':t1-t0}
        dts=M['deep']['ts_ns']
        if kind=='mixed':
            a=arrival['t']-t0; f16=M['u16k']['ts_ns'][0]; res['arrival_ns']=a; res['ttft_16k_ns']=f16-M['u16k']['t_send_ns']; res['prefill_complete_ns_after_arrival']=f16-a
            res.update(MX.deep_metrics(dts,a,f16)); res['common']=MX.window_stats({'deep':dts,'u16k':M['u16k']['ts_ns']})
            res['u16k_server_timings']=M['u16k']['timings']
        else: res.update(MX.deep_metrics(dts)); res['common']=None
        res['deep_server_timings']=M['deep']['timings']; res['goodput_total_tps']=(sum(len(m['ts_ns']) for m in M.values())-len(M))/((max(m['ts_ns'][-1] for m in M.values())-min(m['ts_ns'][0] for m in M.values()))/1e9)
        row.update(elapsed_wall_ns=res['wall_ns'],common_decode_window_ns=(res['common'] or {}).get('window_ns'),committed_tokens_in_common_window=(res['common'] or {}).get('tokens'),status='PASS',exit_code=0,
                   result_summary={k:v for k,v in res.items() if k not in('deep_server_timings','u16k_server_timings')})
        write_atomic(rd+'/result.json',json.dumps(res,indent=1))
        # per-request rows
        ev=str(len(open(rd+'/events.json').read())); 
        for n,s in streams.items():
            m=M[n]; tm=m['timings']; ts=m['ts_ns']; fixture='d0' if n=='deep' else cmd['uncached_segment']; ptok=(len(prompt_d) if n=='deep' else len(p16))
            slot=deep_slot if n=='deep' else other
            tl=srv.logs(since=t_log); shifted=('context shift' in tl.lower()) 
            rq={'run_id':rid,'request_id':rid+'-'+n,'slot_id':slot,'fixture_id':fixture,'prompt_tokens_actual':m['tokens_evaluated'] if m['tokens_evaluated'] is not None else ptok,'prompt_tokens_submitted':ptok,
                'output_tokens_requested':n_deep if n=='deep' else 32,'output_tokens_actual':tm.get('predicted_n'),'stream_events':len(ts),'context_limit':CTX_SLOT,
                'resident_tokens_start':resident0 if n=='deep' else 0,'resident_tokens_end':m['tokens_cached'],'resident_end_semantics':'server tokens_cached at stop (may be 1 below prompt+emitted)',
                'cache_hit_tokens':tm.get('cache_n'),'prompt_eval_tokens':tm.get('prompt_n'),'context_shifted':shifted,'truncated':m['truncated'],
                'ttft_ns':ts[0]-m['t_send_ns'],'decode_elapsed_ns':ts[-1]-ts[0],'ttft_semantics':'client: request send -> first SSE token event; decode_elapsed: first->last token event','completion_status':m['stop_type'] or ('EOS' if m['stopped_eos'] else 'limit' if m['stopped_limit'] else None)}
            append_jsonl(REQS,rq); recs.append(rq)
        print(rid,json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in res.items() if k in('ttft_16k_ns','tps_own_span','itl_p50_ms','itl_p95_ms','itl_max_ms','tps_during_prefill','prefill_complete_ns_after_arrival')}),file=out,flush=True); fails[0]=0
    except Exception as ex:
        rc=1; traceback.print_exc(file=err); row.update(status='FAIL',exit_code=1,missing_reason=repr(ex)); fails[0]+=1
    finally:
        tel[0].terminate(); tel[1].close()
        tl=srv.logs(since=t_log); write_atomic(rd+'/server_log_slice.txt',tl); write_atomic(rd+'/cmd.json',json.dumps(cmd,indent=1)); 
        write_atomic(rd+'/env.txt','\n'.join(ENV[srv.arm])+'\nLD: lib='+LIBS[srv.arm]+' -> /app/libggml-cuda.so\n'); write_atomic(rd+'/rc',str(rc)); write_atomic(rd+'/stdout.txt',out.getvalue()); write_atomic(rd+'/stderr.txt',err.getvalue())
        if buildup_running(): row['contamination'].append('build-up running at end')
        row['end_utc']=iso(); append_jsonl(RUNS,row); print(out.getvalue(),err.getvalue(),flush=True)
    return row

def prefill_fixture(srv,fx):
    """one-time: prefill d0 (98048 tokens) on slot 0, save slot file d0.bin. recorded as run."""
    rid=now_utc()+'-d0-%s-prefill'%srv.arm; rd=E+'/runs/'+rid; os.makedirs(rd,exist_ok=False); row={'schema_version':SCHEMA,'session_id':SESSION_ID,'run_id':rid,'evidence_origin':'new_measurement','variant_id':srv.arm,'workload_id':'e3-fixture-prefill','binary_sha256':sha_file(LIBS[srv.arm]),'instrumented':False,'warmup':True,'start_utc':iso(),'contamination':['build-up running at start'] if buildup_running() else [],'raw_paths':[rid]}
    try:
        p=prompt_for(fx,'d0'); t=time.time(); st,j=http_json('/completion',{'prompt':p,'n_predict':1,'id_slot':0,'cache_prompt':True,'temperature':0,'ignore_eos':True}); assert st==200,j
        tm=j['timings']; st2,j2=http_json('/slots/0?action=save',{'filename':'d0.bin'}); assert st2==200,j2
        write_atomic(rd+'/result.json',json.dumps({'timings':tm,'save':j2,'tokens_evaluated':j.get('tokens_evaluated'),'tokens_cached':j.get('tokens_cached'),'truncated':j.get('truncated'),'prompt_sha256':hashlib.sha256(json.dumps(p).encode()).hexdigest()},indent=1))
        row.update(status='PASS',exit_code=0,elapsed_wall_ns=int((time.time()-t)*1e9),pp_tps=tm.get('prompt_per_second'),prompt_n=tm.get('prompt_n'),note='slot file d0.bin written by this arm and reused for both arms')
    except Exception as ex: row.update(status='FAIL',exit_code=1,missing_reason=repr(ex)); write_atomic(rd+'/stderr.txt',traceback.format_exc())
    row['end_utc']=iso(); append_jsonl(RUNS,row); return row

def session(arm,units,fx,ub=1024,prefill=False,tag=''):
    srv=Server(arm,ub,tag)
    try:
        srv.start(); full,_=fixtures(['d0'],) if False else (None,None)
        rec={'session_id':SESSION_ID,'server_session':srv.sid,'arm':arm,'ub':ub,'start_utc':srv.t_start,'load_s':srv.load_s,'engage':srv.engage,'kv_lines':srv.kv_lines,'lib_sha256':sha_file(LIBS[arm]),'image':IMG,
             'buildup_running':buildup_running(),'gpu_mem':sh(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,memory.total','--format=csv,noheader']).stdout.strip()}
        append_jsonl(SESS,rec)
        if prefill:
            r=prefill_fixture(srv,fx); print('prefill',r.get('status'),r.get('pp_tps'),flush=True)
            if r['status']!='PASS': raise RuntimeError('prefill failed')
        # session warm-up (discarded but kept): alone, 64 tokens
        run_unit(srv,'alone',fx,'warm',0,0,0,True,64,0,0)
        for u in units:
            if fails[0]>=2: raise RuntimeError('2 consecutive failures; abort')
            unit_ctr[0]+=1
            for kind in ('alone','mixed'):
                run_unit(srv,kind,fx,u['cycle'],u['block'],u['pos'],u['deep_slot'],u.get('warm',False),u.get('n_deep',256),unit_ctr[0]-1,u['order'])
    finally:
        srv.stop()
    return srv

def load_fx():
    # needs a live server for /tokenize: use a CPU-light vocab path -> handled by first session (see main)
    pass
if __name__=='__main__':
    plan=sys.argv[1]
    print('start',iso(),flush=True)
    if gpu_busy(): sys.exit('GPU busy: '+gpu_busy())
    unload_35b(); time.sleep(5)
    if gpu_busy(): sys.exit('GPU busy after unload: '+gpu_busy())
    names=['d0','d1','sentinel']+['u%02d'%i for i in range(16)]
    # tokenization needs server: do inside a first short-lived start with arm A (fixtures cached on disk afterwards)
    need=[n for n in names if not os.path.exists(E+'/fixtures/tokens/%s.json'%n)]
    srv0=None
    if need:
        srv0=Server('A',1024,'-tok').start()
        try:
            fx,_=fixtures(names); write_fixture_manifest(fx)
        finally: srv0.stop()
    fx,_=fixtures(names)
    bl=[]; order=0
    def U(cycle,block,pos,slot): 
        global order; order+=1; return {'cycle':cycle,'block':block,'pos':pos,'deep_slot':slot,'order':order}
    try:
        if plan in('smoke','all'):
            session('A',[dict(U('smoke',0,1,0),n_deep=128)],fx,prefill=not os.path.exists(E+'/slots/d0.bin'),tag='-smoke')
        if plan in('blocks','all'):
            seq=[]  # sessions: A | BB | AA | BB | AA | BB | A  (ABBA x3, adjacent same-arm positions share a server session)
            for b in (1,2,3):
                slot=0 if b%2 else 1
                for pos,arm in enumerate('ABBA',1): seq.append((arm,U(1,b,pos,slot) if False else U('c%d'%b,b,pos,slot)))
            groups=[]
            for arm,u in seq:
                if groups and groups[-1][0]==arm: groups[-1][1].append(u)
                else: groups.append((arm,[u]))
            for arm,us in groups: session(arm,us,fx)
    finally:
        sh(['bash','-c','docker rm -f $(docker ps -aq --filter name=^ev-) 2>/dev/null'])
    print('done',iso(),flush=True)
