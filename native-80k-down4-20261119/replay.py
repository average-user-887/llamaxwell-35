"""Verify the exported traces and numerical summaries offline."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from math import isclose

root=Path(__file__).resolve().parent
def read(name):
    return json.loads((root/name).read_text())

def check(mode):
    index=read(f'{mode}-trace-index.json')
    receipt=read(f'{mode}-receipt.json')
    assert receipt['software']==read('profile.json') and receipt['experts']==8
    contents=(root/index['file']).read_bytes()
    rows=[json.loads(line) for line in contents.splitlines()]
    assert sha256(contents).hexdigest()==index['trace_sha256']
    assert Counter(r['event'] for r in rows)==index['events']
    assert all(r['arm']==mode for r in rows)
    assert rows[-1]['runner_exit']==rows[-1]['cleanup_exit']==0
    boundaries=[r['counters'] for r in rows if r['event']=='physical_counter_boundary']
    for p,n,a,b in zip(('cold','warm'),(192,3072),boundaries,boundaries[1:],strict=False):
        phase=receipt['phases'][p]
        record,= [r for r in rows if r['event']=='observed_response_metrics' and r['phase']==p]
        times=phase['timings']
        assert times==record['server_timings']
        assert times['predicted_n']==phase['same_mode_archive']['exact_ids']==n
        assert phase['same_mode_archive']['exact_full_text'] and phase['same_mode_archive']['cache_and_draft_totals_equal']
        assert times['cache_n']==(0 if p=='cold' else 79996)
        assert times['prompt_n']==(80000 if p=='cold' else 4)
        assert isclose(phase['delivered_tokens_per_s'],1000*n/times['predicted_ms'])
        assert isclose(times['predicted_per_second'],1000*(n-1)/times['predicted_ms'])
        expected=40*times.get('draft_n',0)
        assert all(b[k]-a[k]==expected for k in ('down4','combined','skip'))
        assert receipt['down4']['phases'][p]['down4_launches']==expected
        if mode=='native':
            assert [r['output_token'] for r in rows if r['event']=='observed_stream_output_ordinal' and r['phase']==p]==list(range(1,n+1))
        else:
            assert sum(r['emitted'] for r in rows if r['event']=='cycle' and r['phase']==p)==n
    assert len(boundaries)==3 and receipt['semantic_score'] is None
    return receipt

native=check('native')
if (root/'pair.json').exists():
    pair=read('pair.json')
    serial=check('serial')
    for p in ('cold','warm'):
        a,b=(r['phases'][p]['delivered_tokens_per_s'] for r in (serial,native))
        assert isclose(pair['phases'][p]['same_build_gain_pct'],100*(b/a-1))
        assert read('repetition.json')[p]['native']==native['phases'][p]['same_mode_archive']['repetition']
    assert pair['identical_child_flags'] and pair['only_argv_difference_is_speculation']
print('PASS offline receipt replay')
