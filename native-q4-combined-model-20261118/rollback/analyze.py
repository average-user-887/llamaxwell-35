import hashlib, importlib.util, json, re, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
C=HERE.parent
N34=C/'night-gdn-t2-state'
COMBINED=C/'resume-q4-combined-native/libggml-cuda.so'
COMBINED_SHA='abe7dcfdbf2d7df0b31b5478212f3234ffdd8e2b8f9a077c0a216b2e32865aae'
N100=C/'night-native-depth2-q3-80k-r1'
spec=importlib.util.spec_from_file_location('n34',N34/'summarize.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need=base.require
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_report(out):
    modes=['wide_a','wide_b','serial']
    correction=int((out/'correction.id').read_text())
    tape=[int(x) for x in (N34/'inputs/oracle.ids').read_text().split()]
    # Match the actual loaded paths to the precise DSO hashes printed inside the container.
    module_names=['libllama.so','libggml.so','libggml-base.so','libggml-cuda.so']
    need(digest(COMBINED)==COMBINED_SHA,'combined CUDA DSO changed')
    hashes={(COMBINED_SHA if x=='libggml-cuda.so' else digest(N100/'libs'/x))+'  /candidate/'+x for x in module_names}
    hashes.add(digest(HERE/'state')+'  /tool/state')
    console=(out/'container.log').read_text()
    need(hashes<=set(console.splitlines()),'runtime DSO or executable hashes differ')
    arms={}
    def state_delta(a,b,stage,pos):
        return base.compare_state(base.State(out/f'{a}.{stage}.state',pos),base.State(out/f'{b}.{stage}.state',pos))
    def kv_delta(a,b,stage):
        x,y=[np.memmap(out/f'{m}.{stage}.kv','u1','r') for m in (a,b)]
        need(len(x)==len(y)>0,'KV records differ in shape')
        return {'bytes':len(x),'different_bytes':int(np.count_nonzero(x!=y))}
    for m in modes:
        rows=(out/f'{m}.rows').read_text();log=(out/f'{m}.log').read_text()
        need('q4_r2_geometry' not in log,'unexpected long sparse attention on the 2K fixture')
        need(re.findall(r'offloaded (\d+/\d+) layers to GPU',log)==['42/42'],'physical offload')
        expected={str((N100/'libs'/x).resolve()) if x!='libggml-cuda.so' else '/candidate/libggml-cuda.so' for x in module_names}
        need(set(re.findall(r'^ROLL4_MAP (\S+)$',log,re.M))==expected,'actual DSO map differs')
        start,=base.records(rows,'START');end,=base.records(rows,'END')
        need(start['mode']==end['mode']==m and start['rs']=='3' and start['offload']=='42/42' and
             int(start['vocab'])==base.VOCAB and start['excluded']=='5' and end['end']=='2009','context/finish proof')
        states=base.records(rows,'STATE')
        expected_stages=[('prefix',2003)]+([] if m=='serial' else [('badquad',2007)])+[('kept',2004),('corrected',2005)]
        need([(r['stage'],int(r['pos'])) for r in states]==expected_stages and all(int(r['bytes'])==base.PARTIAL_BYTES for r in states),'retained logical state stages')
        kv=re.findall(r'^ROLL4_KV stage=(\w+) pos=(\d+) bytes=(\d+) full_bytes=(\d+)$',rows,re.M)
        need([(s,int(p)) for s,p,_,_ in kv]==expected_stages and all(int(k)>0 and int(f)-int(k)==base.PARTIAL_BYTES for _,_,k,f in kv),'attention/recurrent split proof')
        calls=base.records(rows,'CALL')
        expected_calls=[('prefix',2000+i,1,tape[i],tape[i]) for i in range(4)]+(
          [('kept',2004,1,tape[4],tape[4])] if m=='serial' else [('badquad',2004,4,tape[4],46 if m=='wide_a' else 47)])+[
          ('corrected',2005,1,correction,correction)]+[('follow',2000+i,1,tape[i],tape[i]) for i in range(6,10)]
        need([(r['stage'],int(r['at']),int(r['in']),int(r['input0']),int(r['input_last'])) for r in calls]==expected_calls and
            all(r['graphs']=='1' and r['in']==r['out'] for r in calls),'actual input/output graph widths/tape')
        remove=re.findall(r'^ROLL4_REMOVE start=(\d+) after=(\d+) before=(\d+) wrong=(\d+) middle=(\d+) tail=(\d+) first_choice=(\d+)$',rows,re.M)
        choice=int(end['first_choice']);need(0<=choice<base.VOCAB,'invalid target correction')
        expected_rollback=[] if m=='serial' else [('2005','2004','2007','42' if m=='wide_a' else '44','43' if m=='wide_a' else '45','46' if m=='wide_a' else '47',str(choice))]
        need(remove==expected_rollback and (not remove or choice!=int(remove[0][3])),'first wrong proposal or native rewind missing')
        need(int(end['stock'])+int(end['stock_cache'])==(0 if m=='serial' else 30) and
             int(end['wrong_shape'])==int(end['t2_launches'])==int(end['t2_attempts'])==0,'30 real K4 GDN layers or branch proof missing')
        moe=re.findall(r'^ROLL4_MOE active_q4=(\d+) off_prefill=(\d+) independent_wrong=(\d+)$',rows,re.M)
        need(len(moe)==1 and int(moe[0][0])==(0 if m=='serial' else 40) and
             int(moe[0][2])==(0 if m=='serial' else 3),'native width4 MoE proof or rejected suffix mismatch')
        if m!='serial':
            need(re.search(r'Q4_COHORT residency device=0 sm=24 active_per_sm=[2-9] regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1',log),'resident Q4 cohort device execution missing')
        if m=='wide_a': need(choice==correction,'first wide target did not choose correction')
        arms[m]={'selected_next':choice,'correction_supplied':correction,'correction_is_own_top1':choice==correction,
             'next_after_correction_and_four_teacher_forced_steps':[int(r['next0']) for r in calls if r['stage'] in ('corrected','follow')],
             'native_rollback':remove,'measured_width4_cohort_launches':int(moe[0][0]),'width4_launches_before_rejection':int(moe[0][1]),'forced_suffix_length':int(moe[0][2]),'verified_K4_GDN_stock':int(end['stock']),'verified_K4_GDN_direct_cache':int(end['stock_cache']),
             'attention_record_bytes':{s:int(k) for s,_,k,_ in kv}}
    def saved(m,stage,kind):
        return (out/f'{m}.{stage}.{kind}.sha256').read_text().split()[0]
    prefix=all(len({saved(m,'prefix',k) for m in modes})==1 for k in ('state','kv'))
    need(prefix,'unmatched fresh prefix')
    different_wrong={k:saved('wide_a','badquad',k)!=saved('wide_b','badquad',k) for k in ('state','kv')}
    comparisons={}
    for a,b in [('wide_a','wide_b'),('wide_a','serial')]:
        pair={}
        for stage,pos in [('kept',2004),('corrected',2005)]:
            recurrent=state_delta(a,b,stage,pos); need(recurrent['nonfinite_values_a_plus_b']==0,'nonfinite state')
            pair[stage]={'attention_kv':kv_delta(a,b,stage),'recurrent':recurrent}
        def logits(m,initial):
            f='kept' if m=='serial' else 'badquad'
            return base.logits(out/f'{m}.{f}.logits',1 if m=='serial' else 4)[0] if initial else base.logits(out/f'{m}.corrected.logits',1)[0]
        pair['logits']={label:base.delta(logits(a,keep),logits(b,keep)) for label,keep in [('kept',True),('corrected',False)]}
        need(all(r['nonfinite_a']==r['nonfinite_b']==0 for r in pair['logits'].values()),'nonfinite logits')
        pair['choices_equal']=arms[a]['next_after_correction_and_four_teacher_forced_steps']==arms[b]['next_after_correction_and_four_teacher_forced_steps']
        comparisons[a+'_vs_'+b]=pair
    same=comparisons['wide_a_vs_wide_b']
    parity=(all(different_wrong.values()) and all(same[s]['recurrent']['raw_bitwise_equal'] and
                 same[s]['attention_kv']['different_bytes']==0 for s in ('kept','corrected')) and
            all(v['different_words']==0 for v in same['logits'].values()) and same['choices_equal'])
    return {'measurement_complete':True,'same_shaped_rejected_suffix_isolation_pass':parity,
        'same_fresh_serial_prefill_state_and_attn_kv':prefix,'rejected_tail_states_different':different_wrong,
        'arms':arms,'comparisons':comparisons,
        'long_sparse_attention_exercised':False,'scope':'At 2K the long sparse Q4 attention selector is not entered. One target-only real K4 rejection retaining row zero at position 2004 and discarding 2005–2007, with two independently re-prefilled wrong tails and independent fresh q1 replay. KV contains serializer metadata and active attention K/V bytes. Recurrent records contain the active 30 conv/SSM layers. Dormant planes and the MTP draft-context internals are outside this probe. Cross-shape parity is descriptive.'}

if __name__=='__main__':
    out=Path(sys.argv[1]); result=load_report(out)
    dest=out/'result.json'
    if dest.exists(): need(json.loads(dest.read_text())==result,'saved result differs')
    else: dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('measurement_complete','same_shaped_rejected_suffix_isolation_pass','same_fresh_serial_prefill_state_and_attn_kv','rejected_tail_states_different')}))
