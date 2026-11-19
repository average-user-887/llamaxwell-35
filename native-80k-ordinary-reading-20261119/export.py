"""Project four replayed owner receipts onto an explicit public numerical schema."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from replay import summarize, verify_sources

HERE=Path(__file__).resolve().parent
COUNTS=('cache_n','evaluated_n','raw_generated_n','server_decode_steps','visible_retokenized_n','streamed_id_n','offered','accepted')
TIMES=('prompt_ms','server_decode_ms','first_last_stream_s','token_ttft_s','request_wall_s','visible_ttft_s')
FLAGS=('sustained','all_generated_ids_streamed','declared_word_observed','word_verified_in_raw_ids','task_terminal','truncated')
QUALITY=('choice_correct','choice_total','nonblank_lines','duplicate_lines','word_windows_24','repeated_word_windows_24','book_copied_word_windows_24')
COMPACT=('requested','launched','compact_only','ambiguous','unsupported','raw_bytes','compact_bytes','shadow_rows','shadow_mismatches')
DOWN=('row0','row1','combined','skip','q2_sparse','q2_stride','down4')


def project(d,keys,cls):
    out={}
    for k in keys:
        v=d[k]
        if type(v) is not cls:
            raise ValueError('unexpected type for '+k)
        out[k]=v
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('owner',type=Path)
    ap.add_argument('receipts',type=Path,nargs=4,metavar='RECEIPT')
    ap.add_argument('--destination',type=Path,default=HERE)
    args=ap.parse_args()
    verify_sources()
    owner=args.owner.resolve()
    pins=json.loads((HERE/'source-pins.json').read_text())
    from hashlib import sha256
    if sha256((owner/'FROZEN.sha256').read_bytes()).hexdigest()!=pins['owner_freeze_sha256']:
        raise ValueError('owner freeze differs')
    for k,h in pins['exact_owner_scripts_sha256'].items():
        if sha256((owner/k).read_bytes()).hexdigest()!=h:raise ValueError('owner source differs')
    sys.path.insert(0,str(owner))
    spec=importlib.util.spec_from_file_location('ordinary_owner',owner/'model.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    receipts=[p.resolve() for p in args.receipts]
    actual=mod.compare(receipts)
    runtime=json.loads((HERE/'profile.json').read_text())
    exported=[]
    for slot,(p,a) in enumerate(zip(receipts,actual['arms'],strict=True)):
        if a['mapped']!=runtime or set(a['phases'])!={'cold'}:raise ValueError('runtime, phase or cache design changed')
        status=dict(x.split('=',1) for x in (p/'status.txt').read_text().splitlines())
        manifest=dict(x.split('=',1) for x in (p/'manifest.txt').read_text().splitlines() if '=' in x)
        t=json.loads((p/'terminal.json').read_bytes())
        m=a['phases']['cold']; q=m['quality']
        if q['blind_rationale_grounding']!=[None]*4 or status['warm_requested']!='0':raise ValueError('unexpected task review or warm setting')
        if m['stop_type'] not in ('word','eos','limit','none'):raise ValueError('unknown terminal')
        exp={'order':slot+1,'mode':a['mode'],'experts':a['experts_used'],'start_epoch':int(manifest['start_epoch']),
             'completion_epoch':int(status['completed_epoch']),'cap_seconds':int(manifest['cap_seconds']),
             **project(t,('runner_exit','cleanup_exit'),int),**project(m,COUNTS,int),**project(m,TIMES,float),
             **project(m,FLAGS,bool),'stop_type':m['stop_type'],'quality':{**project(q,QUALITY,int),
             **project(q,('malformed','all_choices_correct'),bool),'blind_rationale_grounding':None},
             'compact':project(m['compact'],COMPACT,int),'down4':project(m['down4_physical_delta'],DOWN,int),
             'guard':project(m['guard'],('ready','observer','types','rollback','shape','model','draft','sampler','mask'),int)}
        # Raw server output and client ID event accounting have independent clocks.
        # The first token-bearing SSE packet in this implementation contains exactly one sampled ID.
        with (p/'cold-events.jsonl').open() as f:
            first=next(x for line in f if (x:=json.loads(line))['tokens'])
        exp['first_stream_packet_id_n']=len(first['tokens'])
        exported.append(exp)
    out={'schema':1,'arms':exported}
    summary=summarize(out)
    args.destination.mkdir(exist_ok=True)
    for filename,obj in [('arms.json',out),('summary.json',summary)]:
        target=args.destination/filename
        temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
        os.replace(temp,target)
    print('Four chronological sealed cold arms exported.')


if __name__=='__main__':main()
