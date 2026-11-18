"""Retain typed phase/size/difference records, including for the first refusal."""
import json,re,sys
from pathlib import Path
def fields(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines())
def keep(row,names):return {k:row[k] for k in names if k in row}
event_keys={
    'N34_START':['rs','offload','prompt','vocab','excluded'],
    'N34_CALL':['stage','at','in','out','graphs'],
    'ROLL4_KV':['stage','pos','bytes','full_bytes'],
    'N34_STATE':['stage','pos','bytes'],
    'ROLL4_MOE':['active_q4','off_prefill','independent_wrong'],
    'ROLL4_REMOVE':['start','after','before'],
    'N34_END':['end','t2_attempts','t2_launches','stock','stock_cache','wrong_shape']}
def export(root):
    status=fields(root/'status.txt');mf=fields(root/'manifest.txt');p=root/'raw'
    result={'format':1,'runtime_cuda_sha256':'abe7dcfdbf2d7df0b31b5478212f3234ffdd8e2b8f9a077c0a216b2e32865aae',
            'runner_exit':int(status['runner_exit']),'cleanup_exit':int(status['cleanup_exit']),'cap_s':int(mf['cap_seconds']),
            'bounded_wall_s':int(status['end_epoch'])-int(mf['start_epoch']),'long_sparse_attention_qualified':False,
            'phase_times_s':None,'timelines':{},'refusals':{},'measured_comparisons':None}
    for mode in ('wide_a','wide_b','serial'):
        rows=p/f'{mode}.rows';log=p/f'{mode}.log'
        if rows.exists():
            records=[]
            for line in rows.read_text(errors='replace').splitlines():
                tag,_,tail=line.partition(' ')
                if tag not in event_keys:continue
                raw=dict(re.findall(r'(\w+)=([^ ]+)',tail))
                rec=keep(raw,event_keys[tag])
                records.append({'event':tag,**{k:int(v) if v.isdigit() else v for k,v in rec.items()}})
            result['timelines'][mode]=records
        if log.exists():
            error=re.findall(r'^N34_ERROR ([A-Za-z0-9 /_,.=-]{1,130})$',log.read_text(errors='replace'),re.M)
            if error:result['refusals'][mode]=error[0]
    f=root/'result.json'
    if f.exists():
        raw=json.loads(f.read_text())
        result.update(keep(raw,['measurement_complete','same_shaped_rejected_suffix_isolation_pass','same_fresh_serial_prefill_state_and_attn_kv','rejected_tail_states_different']))
        numeric=['f32_count','different_words','different_bytes','nonfinite_a','nonfinite_b','max_abs','rms']
        cmp={}
        for name,pair in raw['comparisons'].items():
            rec={'choices_equal':pair['choices_equal'],'logit_row_differences':{k:keep(v,numeric) for k,v in pair['logits'].items()}}
            left,right=name.split('_vs_');rec['first_choice_equal']=raw['arms'][left]['selected_next']==raw['arms'][right]['selected_next']
            for stage in ('kept','corrected'):
                t=pair[stage];state=t['recurrent']
                rec[stage]={'attention_kv':keep(t['attention_kv'],['bytes','different_bytes']),'recurrent':{
                    **keep(state,['raw_bitwise_equal','metadata_bytes','metadata_equal','payload_f32_words','different_f32_words','different_payload_bytes','nonfinite_values_a_plus_b','max_abs','nonidentical_layer_rows']),
                    'per_layer':[keep(row,['layer','kind',*numeric]) for row in state['per_layer']]}}
            cmp[name]=rec
        result['measured_comparisons']=cmp
    (root/'portable-receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':export(Path(sys.argv[1]))
