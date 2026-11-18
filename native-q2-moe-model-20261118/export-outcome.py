"""Export completed model records without stream payload, request identifiers or raw device data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import trace_export as t

KEYS = ('row0', 'row1', 'combined', 'skip')
HERE = Path(__file__).resolve().parent

def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def repetition(text):
    norm = lambda s: re.sub(r'\d+', '#', ' '.join(s.split()))
    lines = [norm(line) for line in text.splitlines() if line.strip()]
    words = re.findall(r'\w+',norm(text))
    tail = lines[3*len(lines)//4:]
    clean = [' '.join(line.split()) for line in text.splitlines() if line.strip()]
    lead = re.compile(r'^(?:(?:async\s+)?(?:def|for|with)|class|if|elif|else|while|try|except|finally|return|raise|yield|from|import|match|case)\b')
    trailing = next((i for i,line in enumerate(reversed(clean)) if not line.startswith('#')),len(clean))
    malformed = sum(ord(c)==0xfffd or (ord(c)<32 and c not in '\r\t\n') for c in text)
    windows = [tuple(words[i:i+24]) for i in range(max(0,len(words)-23))]
    return {'normalization':'collapse whitespace and replace runs of digits; count excess instances above one',
            'nonblank_lines':len(lines),'duplicate_normalized_lines':len(lines)-len(set(lines)),
            'final_quarter_lines':len(tail),'final_quarter_normalized_duplicates':len(tail)-len(set(tail)),
            'python_keyword_led_lines':sum(bool(lead.match(line)) for line in clean),'trailing_comment_only_lines':trailing,
            'malformed_characters':malformed,'word_windows_24':len(windows),'repeated_24_word_windows':len(windows)-len(set(windows))}

def export(raw, label, out):
    mode, rev = label.split('-')
    if mode not in ('serial','native') or rev not in ('r0','r1'):
        raise ValueError('typed arm or revision required')
    if raw.name.split('-')[0] != mode or raw.parent.name!='receipts':
        raise ValueError('use original arm directory')
    status=t.fields(raw/'status.txt')
    assert 'completed_epoch' in status
    rec=t.read_json(raw/'sanitized/model-receipt.json')
    assert rec and rec['mode']==mode
    assert (rec['runner_exit'], rec['cleanup_exit'])==tuple(int(status[k]) for k in ('runner_exit','cleanup_exit'))
    logs=t.read_text(raw/'server.log');run=t.read_text(raw/'run.log')
    startup=t.read_text(raw/'server-start.log')
    health=re.search(r'owned server failed to become healthy within (45|75)s',run)
    full=re.search(r'(\d+\.\d{2}\.\d{3}\.\d{3}) I load_tensors: offloaded (\d+)/(\d+) layers to GPU',logs)
    # These fields describe original observation even if startup validation did not finish.
    readout={'official':rec,'readiness_seconds':int(75 if rev=='r1' else 45),
             'readiness_budget_expired':bool(health),'ready_inventory_available':bool(startup),
             'offload':None if full is None else {'server_clock':full[1],'layers':int(full[2]),'total':int(full[3])},
             'completed_responses':{},'scored_semantic_quality':None,'token_or_text_payload_published':False}
    for phase in ('cold','warm'):
        response,n,requested,reason,complete=t.request(raw,phase,False)
        if not response:continue
        times=response.get('timings') or {}
        readout['completed_responses'][phase]={'requested':requested,'actual':n,'complete':complete,'finish_reason':reason,
          **{k:times.get(k) for k in ('prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second','draft_n','draft_n_accepted')},
          'ttft_s':response.get('ttft_s'),'request_wall_s':response.get('wall_s'),'cache_n':times.get('cache_n'),
          'repetition':repetition(response['text']) if isinstance(response.get('text'),str) else None}
    snapshots=rec['snapshots']; recorded=set()
    def boundary(phase):
        snap=snapshots[phase]
        assert set(snap)==set(KEYS) and all(type(v) is int and v>=0 for v in snap.values())
        recorded.add(phase)
        return {'event':'physical_q2_counter_boundary','arm':mode,'boundary':phase,'counters':snap}
    trace=[]
    for event in t.annotated(mode,raw,1 if mode=='native' else 0):
        phase=event.get('phase')
        if event['event']=='request_begin' and phase=='cold' and 'start' in snapshots:
            trace.append(boundary('start'))
        trace.append(event)
        if event['event']=='trace_metadata':
            trace.append({'event':'offload_evidence','arm':mode,'phase':'startup','offload':readout['offload']})
        if event['event']=='output_boundary' and phase in snapshots:
            trace.append(boundary(phase))
    for phase in ['start','cold','warm']:
        if phase in snapshots and phase not in recorded:
            item=boundary(phase)
            item['trace_placement']='request timeline incomplete; retained in physical snapshot order'
            trace.append(item)
    trace.append({'event':'run_outcome','arm':mode,'runner_exit':rec['runner_exit'],'cleanup_exit':rec['cleanup_exit'],
                  'bounded_wall_s':rec['bounded_wall_s'],'readiness_budget_expired':bool(health),'validated':bool(rec['validated'])})
    path=out/(label+'.trace-v2.jsonl')
    path.write_text(''.join(json.dumps(v,separators=(',',':'),allow_nan=False)+'\n' for v in trace))
    readout['trace_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    readout['trace_event_counts']=dict(sorted(Counter(v['event'] for v in trace).items()))
    write(out/(label+'.receipt.json'),readout)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path);parser.add_argument('label',choices=['native-r0','native-r1','serial-r1'])
    parser.add_argument('destination',type=Path)
    opts=parser.parse_args();export(opts.original,opts.label,opts.destination)
