"""Independently verify source identities and public four-arm numeric arithmetic."""
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def need(ok,msg):
    if not ok:raise ValueError(msg)


def load(name):
    return json.loads((HERE/name).read_text())


def verify_sources():
    pins=load('source-pins.json')
    files={p.name for p in (HERE/'source').iterdir() if p.is_file()}
    rel=pins['relocation']
    expected=set(pins['exact_owner_scripts_sha256'])-{rel['source_name']}|{rel['template_file']}
    need(files==expected,'source inventory changed')
    for name,digest in pins['exact_owner_scripts_sha256'].items():
        if name==rel['source_name']:
            content=(HERE/'source'/rel['template_file']).read_bytes()
            need(sha256(content).hexdigest()==rel['template_sha256'] and
                 content.count(rel['placeholder'].encode())==1,'relocated source template differs')
        else:need(sha256((HERE/'source'/name).read_bytes()).hexdigest()==digest,'source hash differs')
    need(sha256((HERE/'profile.json').read_bytes()).hexdigest()==pins['runtime_profile_sha256'],'runtime pins differ')


def verify_manifest():
    paths=set()
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        h,f=line.split('  ',1)
        need(f not in paths and not Path(f).is_absolute() and '..' not in Path(f).parts,'manifest path')
        paths.add(f)
        need(sha256((HERE/f).read_bytes()).hexdigest()==h,'packet content changed: '+f)
    expected={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
    need(expected==paths,'manifest inventory differs')


def summarize(public):
    rows=public['arms']
    need(public['schema']==1 and [r['order'] for r in rows]==[1,2,3,4] and
         [r['mode'] for r in rows]==['serial','native','native','serial'],'ABBA order')
    need(all(a['completion_epoch']<=b['start_epoch'] for a,b in zip(rows,rows[1:])),'overlapping processes')
    out=[]
    for r in rows:
        need(r['runner_exit']==r['cleanup_exit']==0 and r['cap_seconds']==430 and
             0<r['completion_epoch']-r['start_epoch']<=430,'terminal/lease')
        need(r['experts']==8 and r['evaluated_n']==80000 and r['cache_n']==0,'ordinary cold dimensions')
        n=r['raw_generated_n']; steps=r['server_decode_steps']; d=r['down4']; c=r['compact']
        offered=r['offered']; accepted=r['accepted']; q=r['quality']; guards=r['guard']
        need(1<=n<=1536 and steps==n-1 and r['sustained']==(r['visible_retokenized_n']>=512),'generation units')
        need(0<r['streamed_id_n']<=n and r['all_generated_ids_streamed']==(r['streamed_id_n']==n),'SSE coverage')
        need(r['stop_type'] in {'word','eos','limit','none'} and
             (not r['task_terminal'] or (r['declared_word_observed'] and r['stop_type']=='word' and not r['truncated'])),'stop observation')
        need(r['word_verified_in_raw_ids'] or not (r['all_generated_ids_streamed'] and r['task_terminal']),'word witness')
        need(all(isfinite(r[k]) and r[k]>0 for k in ['server_decode_ms','prompt_ms','first_last_stream_s','token_ttft_s','visible_ttft_s','request_wall_s']),'clocks')
        need(0<=accepted<=offered and d['combined']==d['skip']==d['down4']==40*offered and
             d['row0']==d['row1']==0 and (offered>0)==(r['mode']=='native'),'down4 dispatch')
        need(sum(guards.values())==offered and c['requested']==guards['ready'] and c['launched']<=c['requested']
             and c['compact_bytes']==16*c['launched'] and c['raw_bytes']==1986560*(c['requested']-c['compact_only'])
             and c['compact_only']+c['ambiguous']+c['unsupported']==c['requested']
             and c['shadow_rows']==c['shadow_mismatches']==0,'compact accounting')
        need(q['choice_total']==4 and 0<=q['choice_correct']<=4 and q['all_choices_correct']==(q['choice_correct']==4)
             and q['blind_rationale_grounding'] is None and 0<=q['duplicate_lines']<=q['nonblank_lines']
             and all(0<=q[k]<=q['word_windows_24'] for k in ('repeated_word_windows_24','book_copied_word_windows_24')),'task accounting')
        span=r['first_last_stream_s']
        need(1<=r['first_stream_packet_id_n']<=r['streamed_id_n'] and span+r['token_ttft_s']<=r['request_wall_s']+.000001,'client event timing')
        out.append({'mode':r['mode'],'server_decode_tps':1000*steps/r['server_decode_ms'],
          'delivered_count_over_server_decode_s':1000*n/r['server_decode_ms'],
          'first_to_last_stream_id_tps':(r['streamed_id_n']-r['first_stream_packet_id_n'])/span,
          'visible_whole_request_tps':r['visible_retokenized_n']/r['request_wall_s'],
          'acceptance_fraction':accepted/offered if offered else None,'correct_choices_out_of_four':q['choice_correct'],
          'sustained_visible_length':r['sustained'],'declared_terminal':r['task_terminal']})
    pairs=[]
    for s,nat in [(0,1),(3,2)]:
        ratios={k:out[nat][k]/out[s][k] for k in ('server_decode_tps','delivered_count_over_server_decode_s','first_to_last_stream_id_tps')}
        pairs.append({'serial_arm':s+1,'native_arm':nat+1,'native_over_serial':ratios,
                      'native_minus_serial_cold_request_s':rows[nat]['request_wall_s']-rows[s]['request_wall_s']})
    return {'arms':out,'pairs':pairs,'distinct_tasks':1,'independent_ordered_pairs':2,'agentic_quality':None,'blind_rationale_review':None}


if __name__=='__main__':
    verify_sources()
    if sys.argv[1:]==['--source-only']:
        print('Exact owner source scripts and runtime profile verified.')
    elif not sys.argv[1:]:
        verify_manifest()
        need(summarize(load('arms.json'))==load('summary.json'),'public summary differs')
        print('PASS: packet integrity, four completed chronological arms, numeric witnesses and paired rates.')
    else:raise SystemExit('usage: replay.py [--source-only]')
