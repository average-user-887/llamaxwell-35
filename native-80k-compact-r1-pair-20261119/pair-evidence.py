#!/usr/bin/env python3
"""Export sealed R1 pair aggregates and rederive the public accounting."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
SOURCE_MANIFEST='0046f26bd241019b52ed8d47eecf3c7f57bf6d9bd8979e8de8b53a4ede50ebf7'
PHASES=('cold','warm')
MODES=('serial','native')
TIMING_FIELDS=('cache_n','prompt_n','prompt_ms','prompt_per_token_ms','prompt_per_second','predicted_n','predicted_ms',
               'predicted_per_token_ms','predicted_per_second','draft_n','draft_n_accepted')
CHECKPOINT_FIELDS=('output_token','elapsed_s','event_first_token','event_last_token')
SNAPSHOT_FIELDS=('elided','elided_rejected','kept','restored')
REPETITION_FIELDS=('nonblank_lines','duplicate_normalized_lines','final_quarter_lines','final_quarter_normalized_duplicates',
                   'word_windows_24','repeated_24_word_windows','python_keyword_led_lines','trailing_comment_only_lines','malformed_characters')


def require(ok,message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11)


def phases_from_receipt(receipt, mode, safe_arm):
    snapshot=re.compile(r'^(?:\d+\.\d{2}\.\d{3}\.\d{3} I )?sampler_snapshot: '+
                        ' '.join(k+r'=(\d+)' for k in SNAPSHOT_FIELDS)+'$',re.M)
    expected=(0,)*len(SNAPSHOT_FIELDS)
    result={}
    cold_matches=snapshot.findall((receipt/'server-pre-warm.log').read_text())
    all_matches=snapshot.findall((receipt/'server.log').read_text())
    require(len(cold_matches)==1 and len(all_matches)==2 and cold_matches[0]==all_matches[0],'missing typed sampler snapshots')
    for phase,keys in zip(PHASES,all_matches):
        raw=read(receipt/f'{phase}.json')
        data={key:raw['timings'].get(key) for key in TIMING_FIELDS}
        out=raw['n_out']
        t=data['predicted_ms']
        offers=data['draft_n']
        accepted=data['draft_n_accepted']
        direct={'delivered_tokens':out,'requested_output_tokens':raw['n_predict'],'client_wall_s':raw['wall_s'],'client_ttft_s':raw['ttft_s'],
                'server_timing_fields':data,'stop_reason_limit':raw['finish_reason']=='limit','truncated':raw['truncated']}
        current=tuple(map(int,keys))
        direct['sampler_snapshot_cumulative']=dict(zip(SNAPSHOT_FIELDS,current))
        direct['sampler_snapshot_delta']=dict(zip(SNAPSHOT_FIELDS,(new-old for old,new in zip(expected,current))))
        expected=current
        direct['decode_rates']={'delivered_per_server_decode_s':1000*out/t,'server_reported_post_first_per_s':data['predicted_per_second'],
                                'delivered_per_client_request_wall_s':out/raw['wall_s']}
        if phase=='warm':
            checkpoints=[{k:one[k] for k in CHECKPOINT_FIELDS} for one in raw['observed_sse_checkpoints']]
            require([one['output_token'] for one in checkpoints]==[1,512,1024,1536,2048,2560,3072],'warm SSE checkpoints incomplete')
            segments=[]
            for x,y in zip(checkpoints,checkpoints[1:]):
                dt=y['elapsed_s']-x['elapsed_s']
                n=y['output_token']-x['output_token']
                segments.append({'after_delivered_ordinal':x['output_token'],'through_delivered_ordinal':y['output_token'],
                                 'tokens':n,'elapsed_s':dt,'client_stream_tokens_per_s':n/dt})
            direct['client_stream']={'token_bearing_checkpoints':checkpoints,'segments':segments,'post_first_tokens_per_s':(out-1)/(checkpoints[-1]['elapsed_s']-checkpoints[0]['elapsed_s'])}
        else:
            direct['client_stream']=None
        if mode=='native':
            direct['round_accounting']={'offered':offers,'accepted':accepted,'rejected':offers-accepted,'accepted_fraction':accepted/offers,
                                        'delivered_yield':out/offers,'observed_server_ms_per_offer':t/offers,
                                        'max_ms_per_offer_at_100_delivered_tps':10*out/offers,
                                        'remaining_ms_per_offer_to_100_delivered_tps':(t-10*out)/offers}
        else:
            direct['round_accounting']=None
        direct['repetition_counts']=safe_arm['phases'][phase]['repetition']
        result[phase]=direct
    return result


def rederive(evidence,recon):
    require(evidence['source_freeze_sha256']==recon['source_freeze_sha256'],'source freeze mismatch')
    arms={one['stage']:one for one in evidence['attempts']}
    require(set(arms)==set(MODES) and all(v['disposition']=='validated' for v in arms.values()),'missing exact terminal pair')
    for mode in MODES:
        arm=arms[mode]['model']
        for phase in PHASES:
            row=recon['arms'][mode][phase]
            verified=arm['phases'][phase]
            t=row['server_timing_fields']
            n=row['delivered_tokens']
            elapsed=t['predicted_ms']
            require(n==t['predicted_n']==row['requested_output_tokens']==verified['metrics']['n_out'] and row['stop_reason_limit'] and not row['truncated'],'request terminal counts differ')
            rates=row['decode_rates']
            require(all((close(rates['delivered_per_server_decode_s'],1000*n/elapsed),
                         close(rates['delivered_per_server_decode_s'],verified['metrics']['decode_toks_s']),
                         close(rates['server_reported_post_first_per_s'],1000*(n-1)/elapsed),
                         close(t['predicted_per_second'],rates['server_reported_post_first_per_s']),
                         close(t['predicted_per_token_ms'],elapsed/(n-1)),
                         close(rates['delivered_per_client_request_wall_s'],n/row['client_wall_s']))),'rate convention or exact result differs')
            require(row['repetition_counts']==verified['repetition'] and set(row['repetition_counts'])==set(REPETITION_FIELDS),'repetition counts differ')
            delta=row['sampler_snapshot_delta']
            expected_offers=t['draft_n'] or 0
            rejected=expected_offers-(t['draft_n_accepted'] or 0)
            require(delta=={'elided':expected_offers,'elided_rejected':rejected,'kept':0,'restored':0},'sampler snapshot counters differ')
            compact=verified['compact']
            require(compact['compact_bytes']==16*expected_offers and compact['raw_bytes']==compact['ambiguous']==compact['unsupported']==0,'target compaction bytes or fallback differ')
            round_row=row['round_accounting']
            if mode=='native':
                require(round_row['offered']==expected_offers and round_row['rejected']==rejected and close(round_row['remaining_ms_per_offer_to_100_delivered_tps'],(elapsed-10*n)/expected_offers),'round gate arithmetic differs')
            if phase=='warm':
                stream=row['client_stream']
                checks=stream['token_bearing_checkpoints']
                require(checks[0]['elapsed_s']==row['client_ttft_s'] and checks[-1]['output_token']==n and sum(s['tokens'] for s in stream['segments'])==n-1,'client stream interval counts differ')
                for x,y,s in zip(checks,checks[1:],stream['segments']):
                    require(close(s['elapsed_s'],y['elapsed_s']-x['elapsed_s']) and close(s['client_stream_tokens_per_s'],s['tokens']/s['elapsed_s']),'client interval arithmetic differs')
    for phase in PHASES:
        s,n=(recon['arms'][m][phase] for m in MODES)
        actual=100*(n['decode_rates']['delivered_per_server_decode_s']/s['decode_rates']['delivered_per_server_decode_s']-1)
        require(close(evidence['same_build_pair']['phases'][phase]['same_build_gain_pct'],actual),'strict gain arithmetic differs')
        require(close(recon['pair'][phase]['native_minus_serial_client_wall_s'],n['client_wall_s']-s['client_wall_s']),'client wall delta differs')


def export(source,owner,serial,native,controller_pair):
    require(digest(source/'SHA256SUMS')==SOURCE_MANIFEST,'upstream R1 shadow packet manifest differs')
    checks=load(source/'replay.py','pinned_source_replay')
    pins=checks.sources()
    exporter=load(source/'export-model.py','pinned_source_exporter')
    serial,native=serial.resolve(),native.resolve()
    pair=exporter.export(owner.resolve(),list(zip(MODES,(serial,native))),(serial,native))
    checks.verify_model(pair,pins)
    model=exporter.pin_owner(owner.resolve())
    strict=model.compare(serial,native)
    require(strict==read(controller_pair),'independent owner comparison differs from the controller result')
    payload={'schema':1,'source_freeze_sha256':pair['source_freeze_sha256'],'independent_owner_comparison_equal':True,'arms':{},'pair':{}}
    for mode,receipt,arm in zip(MODES,(serial,native),pair['attempts']):
        payload['arms'][mode]=phases_from_receipt(receipt,mode,arm['model'])
    for phase in PHASES:
        files=[read(folder/f'{phase}.json') for folder in (serial,native)]
        left,right=[v['token_ids'] for v in files]
        first=next((i for i,(a,b) in enumerate(zip(left,right),1) if a!=b),None)
        payload['pair'][phase]={'first_cross_mode_difference_ordinal':first,'cross_mode_full_text_equal':files[0]['text']==files[1]['text'],
                                'native_minus_serial_client_wall_s':files[1]['wall_s']-files[0]['wall_s']}
    rederive(pair,payload)
    write(HERE/'pair-results.json',pair)
    write(HERE/'reconciliation.json',payload)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--r1-source',type=Path)
    p.add_argument('--owner-card',type=Path)
    p.add_argument('--serial',type=Path)
    p.add_argument('--native',type=Path)
    p.add_argument('--controller-pair',type=Path)
    args=p.parse_args()
    inputs=(args.r1_source,args.owner_card,args.serial,args.native,args.controller_pair)
    if all(inputs):
        export(*inputs)
    elif any(inputs):
        p.error('all owner export arguments are required together')
    else:
        rederive(read(HERE/'pair-results.json'),read(HERE/'reconciliation.json'))
        print('R1 terminal split, repetition and rate accounting replayed')
