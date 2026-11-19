"""Run on the private campaign after a terminal arm. Emit allow-listed numeric events."""
import argparse
import importlib.util
import json
import hashlib
from collections import Counter
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('campaign', type=Path)
parser.add_argument('native', type=Path)
parser.add_argument('--serial', type=Path)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('down4_model', args.campaign.resolve()/'break100-model-base/model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
h = m.export
h.ROOT = h.repeat.HERE = ROOT
phases = ('cold', 'warm')


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True)+'\n')


def report(raw, mode):
    raw = raw.resolve()
    validated = m.finished(mode, raw)
    official = m.read(raw/'sanitized/model-receipt.json')
    status = dict(h.base.STATUS.findall((raw/'status.txt').read_text()))
    status.update(h.base.START.findall((raw/'manifest.txt').read_text()))
    status = {k:int(v) for k,v in status.items()}
    assert status['start_epoch'] <= status['completed_epoch'] <= status['deadline_epoch']
    responses = {p:m.read(raw/f'{p}.json') for p in phases}
    counters = {k:h.repeat.numbers(m.read(raw/'witness'/f'{k}.json'), h.repeat.COUNTERS|{'down4'})
                for k in ('start', *phases)}
    assert all(len(x)==7 for x in counters.values())
    info = dict(mode=mode, status=status, terminal=h.repeat.failure_metadata(raw,official,'passed_original_terminal_validator'),
                validated=validated, responses=responses, snapshots=counters)
    trace = h.trace(raw,info)
    old=ROOT/trace['file']
    trace['file']=f'{mode}.trace.jsonl'
    events=[json.loads(line) for line in old.read_text().splitlines()]
    residency=re.compile(r'^Q2_DOWN4 residency device=(\d+) sm=(\d+) active=(\d+) regs=(\d+) static=(\d+) dynamic=(\d+) local=(\d+) ready=(\d+)$')
    fields=('device','sm','active','regs','static','dynamic','local','ready')
    segments,_=h.repeat.phases(raw,False)
    for phase,log in segments.items():
        for line_number,line in enumerate(log.splitlines(),1):
            if matched:=residency.fullmatch(line):
                event=dict(event='down4_residency',arm=mode,phase=phase,relative_log_line=line_number,
                           **dict(zip(fields,map(int,matched.groups()))))
                index=next((i for i,x in enumerate(events) if x.get('phase')==phase and
                            x.get('relative_log_line',0)>line_number),None)
                if index is None:
                    assert phase=='startup'
                    index=next(i for i,x in enumerate(events) if x['event']=='physical_counter_boundary')
                events.insert(index,event)
    data=''.join(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for x in events).encode()
    (ROOT/trace['file']).write_bytes(data)
    old.unlink()
    trace['events']=dict(Counter(x['event'] for x in events))
    trace['trace_sha256']=hashlib.sha256(data).hexdigest()
    assert trace['events'].get('down4_residency',0)==int(counters['warm']['down4']>0)
    write(f'{mode}-trace-index.json',trace)
    stats = {'mode':mode,'software':validated['actual_server_software_sha256'],'down4':validated['down4'],
             'experts':validated['accepted_experts'],'phases':{},'semantic_score':None}
    feature_flags={}
    for entry in (raw/'child-selected-environment.txt').read_text().splitlines():
        match=re.fullmatch(r'((?:GGML_|LLAMA_)[A-Z0-9_]+)=([0-9]+)',entry)
        if match:
            feature_flags[match[1]]=int(match[2])
    assert feature_flags['GGML_CUDA_OCCAMY_MOE_Q2_DOWN4']==1
    stats['numeric_child_feature_flags']=feature_flags
    child_args=(raw/'child-server.args').read_text().splitlines()
    spec=child_args[child_args.index('--spec-type'):]
    expected=['--spec-type','none'] if mode=='serial' else ['--spec-type','draft-mtp','--spec-draft-n-max','1',
                                      '--spec-draft-type-k','f16','--spec-draft-type-v','f16']
    assert spec==expected
    stats['speculation_arguments']=spec
    for p in phases:
        timings = h.repeat.numbers(responses[p]['timings'], set(h.repeat.TIMINGS) | {'predicted_per_second'})
        assert timings['predicted_n'] == len(responses[p]['token_ids'])
        stats['phases'][p] = {'timings':timings, 'delivered_tokens_per_s': validated['metrics'][p]['decode_toks_s'],
                              'same_mode_archive':validated['same_mode_archive'][p]}
    write(f'{mode}-receipt.json',stats)
    return info

native=report(args.native,'native')
if args.serial:
    pair=m.compare(args.serial.resolve(), args.native.resolve())
    serial=report(args.serial,'serial')
    assert native['status']['completed_epoch'] <= serial['status']['start_epoch']
    write('pair.json',{'identical_software':pair['same_build'],'identical_child_flags':True,
        'only_argv_difference_is_speculation':True,'phases':pair['phases'],
        'semantic_score':None,'exact_same_mode_replay':True})
    write('repetition.json',{p:{'serial':pair['arms']['serial']['same_mode_archive'][p]['repetition'],
        'native':pair['arms']['native']['same_mode_archive'][p]['repetition'],
        'cross_mode':h.base.parity(serial['responses'][p], native['responses'][p])} for p in phases})
