"""Replay all published event-pair calculations without a GPU or private fixture."""
import json
import re
from pathlib import Path
from statistics import mean
base=Path(__file__).resolve().parent
r=json.loads((base/'r1/result.json').read_text())
before=json.loads((base/'r0/result.json').read_text())
assert before['complete_order_reversed_pairs']==0 and before['candidate_timing_us'] is None
assert (base/'r0/a1-events.txt').read_text().count('N82_EVENT')==30
assert (base/'r0/b1-events.txt').read_text().count('N82_EVENT')==0
assert 'GGML_ASSERT(' in (base/'r0/b1-events.txt').read_text()
for name, arm in r['arms'].items():
    text=(base/'r1'/(name+'-events.txt')).read_text()
    rows=re.findall(r'^N82_EVENT specialization=(\d+) grid=72 slot=(\d+) tensor=blk\.(\d+)\.attn_qkv\.weight us=([\d.]+)$',text,re.M)
    assert [(int(s),int(i),int(l)) for s,i,l,_ in rows]==[(arm['specialization'],i,l) for i,l in enumerate(r['layers'])]
    ev=[float(t) for *_,t in rows]
    assert ev==arm['kernel_event_us'] and abs(mean(ev)-arm['mean_us'])<1e-7
    assert arm['full_states_logits_choices_match'] and 'fallback0=0 fallback1=0 paired=40 skipped=40' in text
    copies=re.findall(r'^Q8PA_COPY tensor=blk\.(\d+)\.attn_qkv\.weight bytes=(\d+) copy_ms=([\d.]+) resident_bytes=(\d+)$',text,re.M)
    assert sum(int(b) for _,b,_,_ in copies)==arm['allocated_mirror_bytes']
    assert [float(ms) for _,_,ms,_ in copies]==arm['pack_copy_ms_per_layer']
    if name.startswith('b'):
        assert len(copies)==30 and [int(l) for l,*_ in copies]==r['layers']
        assert 'Q8PA_PREPARED before_context=1 layers=30 tensors=120 bytes=426885120' in text
        assert float(re.search(r'Q8PA_PREPARED .*wall_ms=([\d.]+)',text).group(1))==arm['total_preparation_wall_ms']
for i,(a,b) in enumerate([('a1','b1'),('a2','b2')]):
    d=[x-y for x,y in zip(r['arms'][a]['kernel_event_us'],r['arms'][b]['kernel_event_us'])]
    assert d==r['timing']['per_layer_pair_savings_us'][i]
    assert abs(mean(d)-r['timing']['mean_saving_us_per_layer'][i])<1e-8
assert not r['timing']['solo_both_pairs_ge28'] and not r['timing']['both_pairs_ge21_for_conditional_combination']
print('Verified the incomplete first attempt, 150 named kernel events, 60 preparation records, both paired differences and both gate outcomes.')
