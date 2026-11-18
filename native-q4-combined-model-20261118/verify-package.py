import hashlib,json,collections
from pathlib import Path
from warm_compare import compare
p=Path(__file__).resolve().parent
read=lambda q:json.loads(q.read_text())
h=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
for f,r in read(p/'source-pins.json').items():assert h(p/f)==r['portable_sha256']
arms={}
for m in ['smoke-off','smoke-on','serial-on','depth3-off','depth3-attn','depth3-combined']:
 r=arms[m]=read(p/(m+'.receipt.json')); d=p/'traces'/m;t=read(d/'trace-v2-meta.json');lines=d/'trace-v2.jsonl'
 assert h(lines)==t['trace_sha256'] and h(p/'trace_export.py')==t['converter_sha256']
 assert dict(collections.Counter(json.loads(x)['event'] for x in lines.open()))==t['events']
 assert r['status']['runner_exit']==t['runner_exit']==0 and r['status']['cleanup_exit']==t['cleanup_exit']==0
 for v in r['requests'].values():
  assert v['complete_request'] and v['finish_reason']=='limit' and not v['truncated'] and v['eog_token_occurrences']==v['eos_token_occurrences']==0
  k=v['moe_dispatch'];assert sum(k['prefill_layer_launches'])+sum(k['generation_layer_launches'])==k['physical_q4_launches']
 if m.startswith('smoke'): assert r['semantic_score']['passed'] and r['semantic_score']['finish_reason']=='eos'
for l,a,b in [('attention','depth3-off','depth3-attn'),('moe','depth3-attn','depth3-combined')]:
 assert compare(p/'traces'/a,p/'traces'/b)==read(p/(l+'-host-deltas.json'))
for l,a,b in [('attention_only','depth3-off','depth3-attn'),('moe_over_attention','depth3-attn','depth3-combined'),('combined_over_off','depth3-off','depth3-combined'),('strict_native_over_serial','serial-on','depth3-combined')]:
 for f in ('cold','warm'):assert abs(arms[b]['requests'][f]['decode_toks_s']/arms[a]['requests'][f]['decode_toks_s']-read(p/'comparisons.json')['pairs'][l][f]['decode_ratio'])<1e-12
assert arms['depth3-combined']['requests']['warm']['moe_dispatch']['generation_layer_launches']==[949]*40
rb=read(p/'rollback/observed.json'); assert rb['same_shaped_rejected_suffix_isolation_pass'] and rb['measurement_complete'] and rb['runner_exit']==rb['cleanup_exit']==0
for pair,v in rb['measured_comparisons'].items():
 for f in ('kept','corrected'):assert len(v[f]['recurrent']['per_layer'])==60
print("All six outcomes and full chronologies, both matched host comparisons, four cold and warm ratios, physical counts and rollback evidence verified.")
