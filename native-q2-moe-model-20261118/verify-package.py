"""Replay the model packet from the public, count-only chronology and software pins."""
from collections import Counter
import hashlib
import json
from math import isclose
from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
def read(name):return json.loads((p/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    manifest=(p/'SHA256SUMS').read_text().splitlines()
    entries={line.split('  ',1)[1]:line.split('  ',1)[0] for line in manifest}
    assert len(entries)==len(manifest)
    assert set(entries)=={file.name for file in p.iterdir() if file.is_file() and file.name!='SHA256SUMS'}
    assert all(sha(p/name)==digest for name,digest in entries.items())
    allowed=set(__import__('re').findall(r'`([A-Za-z0-9_.-]+)`',(p/'MANIFEST.md').read_text()))
    assert set(entries)=={n for n in allowed if len(n)!=40}
    assert {'native-r0.receipt.json','native-r1.receipt.json','serial-r1.receipt.json','comparison.json'}<=set(entries)
    pins=read('source-pins.json')
    for name,actual in [('supervise.c','supervise.c'),('initial.release.sh','release.sh'),('generate-release.py','generate-release.py'),
                        ('model.py','model.py'),('check-layout.py','check-layout.py'),('initial.FROZEN.sha256','FROZEN.sha256')]:
        assert sha(p/name)==pins['initial_source_sha256'][actual]
    assert sha(p/'recovery.FROZEN.sha256')==pins['recovery_frozen']
    exact=(p/'initial.release.sh').read_bytes()
    fixed=exact.replace(b'started + 45',b'started + 75').replace(b'within 45s',b'within 75s')
    assert fixed!=exact and hashlib.sha256(fixed).hexdigest()==pins['recovery_release']
    profiles=read('profiles.json')
    x,y=profiles['native'],profiles['serial']
    flags=x['environment']
    assert flags==y['environment'] and len(flags)==len(set(flags))
    assert {'GGML_CUDA_OCCAMY_MOE_Q2_COHORTS=1','GGML_CUDA_OCCAMY_MOE_Q2_COUNT=1','GGML_CUDA_OCCAMY_MOE_Q2_READ_IDS=0'}<=set(flags)
    a,b=x['server_arguments'],y['server_arguments']
    assert a[:a.index('--spec-type')]==b[:-2] and b[-2:]==['--spec-type','none']
    recs={}
    for file in sorted(p.glob('*.receipt.json')):
        label=file.name.split('.')[0]
        rec=read(file.name); trace=p/(label+'.trace-v2.jsonl')
        events=[json.loads(line) for line in trace.read_text().splitlines()]
        assert sha(trace)==rec['trace_sha256'] and Counter(e['event'] for e in events)==rec['trace_event_counts']
        assert all(e['arm']==label.split('-')[0] for e in events) and events[-1]['event']=='run_outcome'
        official=rec['official']; assert events[-1]['runner_exit']==official['runner_exit']
        assert events[-1]['cleanup_exit']==official['cleanup_exit'] and rec['scored_semantic_quality'] is None
        data=official['validated']
        if data:
            assert (official['runner_exit'],official['cleanup_exit'])==(0,0) and data['technical_pass']
            assert data['actual_server_software_sha256']['libggml-cuda.so']==pins['candidate_cuda']
            snapshots=data['phase_counter_evidence']['raw_boundaries']
            emission=[v for v in events if v['event']=='output_boundary']
            mode=official['mode']
            assert [v['observed_output_count'] for v in emission]==[192,3072]
            assert all(v['finish_observed'] and v['limit_stop_observed'] for v in emission)
            for i,phase in enumerate(('cold','warm')):
                n=data['metrics'][phase]['draft_n'] or 0
                delta={k:snapshots[i+1][k]-snapshots[i][k] for k in ('row0','row1','combined','skip')}
                assert delta=={'row0':0,'row1':0,'combined':40*n,'skip':40*n}
                assert delta==data['phase_counter_evidence']['phases'][phase]['physical_deltas']
                assert bool(n)==(mode=='native')
                if mode=='native': assert all(data['outputs_vs_archived_q8_native'][phase][k] for k in ('all_ids_equal','full_text_equal'))
        recs[label]=rec
    r0=recs['native-r0']; assert r0['readiness_budget_expired'] and not r0['ready_inventory_available']
    assert r0['official']['runner_exit']==2 and r0['official']['cleanup_exit']==0 and r0['offload']['layers']==42 and not r0['completed_responses']
    if (p/'historical.json').exists():
        import historical
        assert historical.compare()==read('historical.json')
    if (p/'comparison.json').exists():
        pair=read('comparison.json');a=recs['serial-r1']['official']['validated'];b=recs['native-r1']['official']['validated']
        assert a and b and pair['same_cohort_flags'] and pair['same_candidate_cuda_sha256']==pins['candidate_cuda']
        for ph in ('cold','warm'):
            x=a['metrics'][ph];y=b['metrics'][ph];t=pair['measured'][ph]
            assert t['serial']==x and t['native']==y and isclose(t['native_over_serial_tps'],y['decode_toks_s']/x['decode_toks_s'])
            assert isclose(x['request_wall_s']-y['request_wall_s'],t['serial_minus_native_wall_s'])
        assert all(n['strict_first_id_difference_1_based']==170 for n in pair['measured'].values())
        assert pair['strict_warm_2x']==(pair['measured']['warm']['native_over_serial_tps']>=2)
    print('Verified explicit file set, immutable source revisions, typed event chronology and every available terminal arm.')
if __name__=='__main__':verify()
