"""Recompute anchored timing and audit all six chronological public receipts."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import sha256
import json,subprocess
from warm_compare import compare
root=Path(__file__).resolve().parent
ARMS=('smoke-off','smoke-on','serial-off','depth2-off','depth2-on','serial-on')
def read(p):return json.loads(p.read_text())
def check(v,why):
    if not v:raise ValueError(why)
def main():
    rows={};hash_export=sha256((root/'trace_export.py').read_bytes()).hexdigest()
    for name in ARMS:
        receipt=read(root/(name+'.receipt.json'));rows[name]=receipt
        path=root/(name+'.trace-v2.jsonl');meta=read(root/(name+'.trace-v2-meta.json'))
        check(sha256(path.read_bytes()).hexdigest()==meta['trace_sha256'] and hash_export==meta['converter_sha256'],'trace or converter digest')
        counts=Counter();whole=0
        for line in path.open():
            r=json.loads(line);counts[r['event']]+=1
            if r['event']=='host_counter_snapshot':whole+=1
        check(dict(counts)==meta['events'],'chronology length')
        check(meta['runner_exit']==receipt['original_status']['runner_exit'] and meta['cleanup_exit']==0,'status')
        if name.startswith('smoke'):check(receipt['arithmetic']['passed'] and receipt['arithmetic']['finish_reason']=='eos' and counts['arithmetic_score']==1,'short arithmetic')
        for p in ('cold','warm'):
            data=receipt['requests'][p]
            check(data['finish_reason']=='limit' and data['eog_token_occurrences']==data['eos_token_occurrences']==0 and not data['truncated'],'performance stream')
            d=data['moe_dispatch'];check(sum(d['per_target_layer_0_to_39'])==d['kernel_confirmed_launches'],'layer accounting')
    check(rows['depth2-off']['original_status']['runner_exit']==1 and rows['depth2-off']['reconstruction_validated'],'preserved failure/recovery')
    check(rows['depth2-on']['requests']['warm']['moe_dispatch']['per_target_layer_0_to_39']==[1174]*40,'warm coverage')
    check(rows['depth2-on']['requests']['cold']['moe_dispatch']['per_target_layer_0_to_39']==[71]*40,'cold coverage')
    comp=read(root/'comparisons.json')
    for pair,phases in comp.items():
        for phase,v in phases.items():
            a=rows[v['base_arm']]['requests'][phase];b=rows[v['candidate_arm']]['requests'][phase]
            check(abs(v['candidate_ratio']-b['decode_toks_s']/a['decode_toks_s'])<1e-12,'ratio')
    with TemporaryDirectory() as tmp:
        p=Path(tmp)
        for name in ('depth2-off','depth2-on'):
            d=p/name;d.mkdir()
            for f in ('trace-v2-meta.json','trace-v2.jsonl'):(d/f).symlink_to(root/(name+'.'+f))
        check(compare(p/'depth2-off',p/'depth2-on')==read(root/'q3-warm-comparison.json'),'direct warm snapshots')
        for rev in ('initial','r1','r2'):subprocess.run(['python3','-B',str(root/'assemble-source.py'),rev,str(p/rev)],check=True)
    print('Six original chronologies, recovery, launch coverage, strict ratios, complete-counter comparison and three source revisions verified.')
if __name__=='__main__':main()
