"""Reconcile the same IDs, rate convention and sampled host rounds on the archived Q8A native reference."""
import json
from pathlib import Path
from math import isclose

p=Path(__file__).resolve().parent
baseline=p.parent/'native-q8-on-serial-20261118'
def load(path):return json.loads(path.read_text())
def host(trace):
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    h=[row for row in rows if row['event']=='host_counter_snapshot' and row['phase']=='warm']
    assert len(h)>=2 and all(h[i]['rounds']<h[i+1]['rounds'] for i in range(len(h)-1))
    return h
def compare():
    ref=load(baseline/'native-on.reference.json')['measurements']
    new=load(p/'native-r1.receipt.json')['official']['validated']
    assert all(new['outputs_vs_archived_q8_native'][ph]['all_ids_equal'] for ph in ['cold','warm'])
    data={}
    for ph,x in ref.items():
        y=new['metrics'][ph]
        for s in (x,y):assert isclose(s['n_out']/s['decode_s'],s['decode_toks_s'])
        assert (x['n_out'],x['draft_n'],x['draft_n_accepted'])==(y['n_out'],y['draft_n'],y['draft_n_accepted'])
        data[ph]={'historical_decode_rate':x['decode_toks_s'],'candidate_decode_rate':y['decode_toks_s'],
                  'same_convention_rate_ratio':y['decode_toks_s']/x['decode_toks_s'],
                  'historical_request_wall_s':x['request_wall_s'],'candidate_request_wall_s':y['request_wall_s'],
                  'historical_ms_per_offered_round':1000*x['decode_s']/x['draft_n'],
                  'candidate_ms_per_offered_round':1000*y['decode_s']/y['draft_n']}
    old,now=host(baseline/'native-on.trace-v2.jsonl'),host(p/'native-r1.trace-v2.jsonl')
    assert [(r['rounds'],r['accepted_prefix_histogram']) for r in old]==[(r['rounds'],r['accepted_prefix_histogram']) for r in now]
    span=now[-1]['rounds']-now[0]['rounds']
    pieces={}
    for key in old[0]['reported_costs']:
        def diff(s,field):return s[-1]['reported_costs'][key][field]-s[0]['reported_costs'][key][field]
        orig,cand=diff(old,'cumulative_ms')/span,diff(now,'cumulative_ms')/span
        pieces[key]={'historical_ms_per_round_in_window':orig,'candidate_ms_per_round_in_window':cand,'saving_ms_per_round_in_window':orig-cand,
                'historical_calls_in_window':diff(old,'calls'),'candidate_calls_in_window':diff(now,'calls')}
    return {'historical_public_commit':'6d91e01c752c025b7aa11cb4592cbc62269dcba5','matched_decode_rate_definition':'all response IDs divided by reported decode seconds',
            'phases':data,'warm_first_round_snapshot':now[0]['rounds'],'warm_last_round_snapshot':now[-1]['rounds'],
            'warm_complete_rounds_in_window':span,'matching_cumulative_histograms':len(old),'matched_window_host_components':pieces,
            'scope':'Distinct process measurements. Host target time encompasses all target work, without kernel-exclusive attribution.'}
if __name__=='__main__':
    print(json.dumps(compare(),indent=2,allow_nan=False))
