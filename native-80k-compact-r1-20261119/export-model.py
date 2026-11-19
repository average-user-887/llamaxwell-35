#!/usr/bin/env python3
"""Revalidate owner receipts and emit aggregate, allowlisted portable evidence."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile

HERE = Path(__file__).resolve().parent
STAGES = ('shadow','native','serial')
PHASES = ('cold','warm')
METRICS = ('n_out','decode_toks_s','request_wall_s','ttft_s','decode_s','cache_n','evaluated_prompt_n','draft_n','draft_n_accepted','decode_ms_per_round','truncated')
REPETITION = ('nonblank_lines','duplicate_normalized_lines','final_quarter_lines','final_quarter_normalized_duplicates','word_windows_24','repeated_24_word_windows','python_keyword_led_lines','trailing_comment_only_lines','malformed_characters')
COMPACT = ('requested','launched','compact_only','ambiguous','unsupported','raw_bytes','compact_bytes','shadow_rows','shadow_mismatches')
DOWN4 = ('down4_launches','expected','ordinary_q2_launches','ordered_full_experts')
GUARDS = ('ready','observer','types','rollback','shape','model','draft','sampler','mask')
Q2 = ('combined','row0','row1','skip','q2_sparse','q2_stride','down4')
PARITY = ('exact_ids','exact_full_text','cache_and_draft_totals_equal')
PROFILE = ('exe','libggml-cuda.so','libllama-common.so.0.6.0','libllama-server-impl.so','libllama.so.0.6.0')


def read(path):
    return json.loads(path.read_text())


def scalar(value):
    return value if type(value) in (bool,int) or (type(value) is float and math.isfinite(value)) else None


def subset(data, fields):
    return {key:scalar(data.get(key)) for key in fields}


def pin_owner(card):
    pins=read(HERE/'source-pins.json')
    expected={'FROZEN.sha256':pins['owner_source_freeze_sha256'],'model.py':pins['exact_composite_sources']['model.py']}
    for name,digest in expected.items():
        if hashlib.sha256((card/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('owner source changed since staging')
    spec=importlib.util.spec_from_file_location('compact_owner_model',card/'model.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.profile!=read(HERE/'profile.json'):
        raise ValueError('owner runtime profile differs from the staged profile')
    return module


def terminal_status(stage, receipt, card, model):
    if receipt.parent!=card/'receipts' or not receipt.name.startswith(stage+'-'):
        raise ValueError('receipt path is outside the specified owned stage')
    status=(receipt/'status.txt').read_text()
    values={}
    for name in ('runner_exit','cleanup_exit'):
        result=re.findall(rf'^{name}=([0-9]{{1,3}})$',status,re.M)
        if len(result)!=1 or int(result[0])>255:
            raise ValueError('receipt has no typed terminal runner and cleanup outcome')
        values[name]=int(result[0])
    lines=(receipt/'manifest.txt').read_text().splitlines()
    expected=model.q2.n9.hash_file(card/'FROZEN.sha256')+'  '+str(card/'FROZEN.sha256')
    if not lines or lines[0]!='stage='+stage or expected not in lines:
        raise ValueError('receipt does not bind to its stage and frozen sources')
    cap=[line for line in lines if line.startswith('cap_seconds=')]
    if cap!=['cap_seconds=430']:
        raise ValueError('owned receipt cap differs')
    values['cap_seconds']=430
    if (receipt/'observations.sha256').is_file():
        model.hybrid.validator.sha_check(receipt,'observations.sha256')
        values['observation_manifest_verified']=True
    else:
        values['observation_manifest_verified']=False
    return values


def selected_profile(data):
    expected=read(HERE/'profile.json')
    if data!=expected or set(data)!=set(PROFILE):
        raise ValueError('unexpected mapped identity in export')
    if not all(re.fullmatch(r'[a-f0-9]{64}',v) for v in data.values()):
        raise ValueError('malformed artifact identity')
    return data


def safe_record(verified):
    return {'mapped_software_sha256':selected_profile(verified['actual_server_software_sha256']),
            'semantic_score':None,'accepted_experts':scalar(verified.get('accepted_experts')),
            'phases':{
                phase:{'metrics':subset(verified['metrics'][phase],METRICS),
                       'compact':subset(verified['compact'][phase],COMPACT),
                       'guard':subset(verified['compact_guard'][phase],GUARDS),
                       'down4':subset(verified['down4']['phases'][phase],DOWN4),
                       'archived_parity':subset(verified['same_mode_archive'][phase],PARITY),
                       'repetition':subset(verified['same_mode_archive'][phase]['repetition'],REPETITION)} for phase in PHASES},
            'physical_cumulative_snapshots':[subset(value,Q2) for value in verified['phase_counter_evidence']['raw_boundaries']]}


def observations(receipt):
    seen={}
    for phase in PHASES:
        f=receipt/f'{phase}.json'
        if f.is_file():
            obj=read(f)
            seen[phase]=subset(obj,('n_out','wall_s','ttft_s','truncated'))
    return seen


def export_attempt(model, card, stage, receipt, alias):
    receipt=receipt.resolve()
    status=terminal_status(stage,receipt,card,model)
    result={'attempt':alias,'stage':stage,'terminal':status,'model':None,'private_request_observations_excluded':True}
    if status['observation_manifest_verified']:
        result['available_response_counts'] = observations(receipt)
        official=receipt/'sanitized/model-receipt.json'
        result['bounded_wall_s']=scalar(read(official).get('bounded_wall_s')) if official.is_file() else None
    else:
        result['available_response_counts']={}
        result['bounded_wall_s']=None
    if status['cleanup_exit']:
        result['disposition']='cleanup_nonzero'
    elif status['runner_exit']:
        result['disposition']='runner_nonzero'
    elif not status['observation_manifest_verified']:
        result['disposition']='terminal_manifest_missing'
    else:
        try:
            verified=model.finished(stage,receipt)
        except (ValueError, KeyError, OSError):
            result['disposition']='terminal_replay_refused'
        else:
            result.update(disposition='validated',model=safe_record(verified))
    return result


def compare(model, serial, native, chosen_attempts):
    if any(chosen_attempts.get(p.resolve(),{}).get('disposition')!='validated' for p in (serial,native)):
        raise ValueError('strict comparison requires the selected validated native and serial attempts')
    result=model.compare(serial.resolve(),native.resolve())
    return {'serial_attempt':chosen_attempts[serial.resolve()]['attempt'],
            'native_attempt':chosen_attempts[native.resolve()]['attempt'],
            'same_build':selected_profile(result['same_build']),
            'same_process_feature_flags':result['same_process_feature_flags'] is True,
            'semantic_score':None,
            'phases':{phase:subset(result['phases'][phase],('same_build_gain_pct','native_saved_ms_per_round_vs_369c','observed_100_tps')) for phase in PHASES}}


def export(card, receipts, pair):
    model=pin_owner(card)
    seen=defaultdict(int)
    chosen={}
    for stage,path in receipts:
        if stage not in STAGES:
            raise ValueError('unknown stage')
        path=Path(path).resolve()
        if path in chosen:
            raise ValueError('duplicate terminal receipt')
        seen[stage]+=1
        chosen[path]=export_attempt(model,card,stage,path,f'{stage}-{seen[stage]:02d}')
    return {'schema':1,'source_freeze_sha256':read(HERE/'source-pins.json')['owner_source_freeze_sha256'],
            'attempts':list(chosen.values()),'unrepresented_stages':[stage for stage in STAGES if not seen[stage]],
            'same_build_pair':compare(model,*pair,chosen) if pair else None}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner-card',type=Path,required=True)
    parser.add_argument('--receipt',nargs=2,metavar=('STAGE','RECEIPT'),action='append',required=True)
    parser.add_argument('--pair',nargs=2,type=Path,metavar=('SERIAL','NATIVE'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    card=args.owner_card.resolve()
    if output.is_relative_to(card):
        parser.error('keep portable output separate from owner inputs and receipts')
    payload=export(card,args.receipt,args.pair)
    with tempfile.NamedTemporaryFile('w',dir=output.parent,delete=False) as f:
        temp=Path(f.name)
        json.dump(payload,f,indent=2,allow_nan=False)
        f.write('\n')
    os.replace(temp,output)
