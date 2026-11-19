"""Replay local input, phase, task and physical execution observations."""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from fixture import CARD, PRIVATE, STOP, IDS, digest, need, packed, read

parent = CARD.parent/'break100-compact-model-r2'
spec = importlib.util.spec_from_file_location('ordinary_compact_parent',parent/'model.py')
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
fixture = read(CARD/'fixture.json')
expected = {'temperature':0,'seed':7,'n_predict':1536,'ignore_eos':True,'stop':[STOP],
            'id_slot':0,'cache_prompt':True,'stream':True,'return_tokens':True}

def write(path,obj):
    path.write_bytes(packed(obj)+b'\n')


def score(text):
    # Separate strict machine-graded choices from blind source-grounding review.
    strict=list(re.finditer(r'(?m)^[ \t]*(150|151|152|153)[ \t]+([A-D])\.[ \t]+',text))
    # Score an unambiguous marked option even when the answer uses Markdown headings.
    hdr=r'(?m)^[ \t]*(?:#{1,6}[ \t]+|[-*][ \t]+)?(?:\*\*)?(150|151|152|153)[.):]?[ \t]+(?:\*\*)?\(?([A-D])(?:[.):]|(?=\*\*))(?:\*\*)?[ \t]*(?:[-–—][ \t]*)?'
    matches=list(re.finditer(hdr,text))
    gold=read(PRIVATE/'gold.json')
    groups={}
    order=[]
    for n,m in enumerate(matches):
        key=m[1]
        rationale=text[m.end():(matches[n+1].start() if n+1<len(matches) else len(text))].strip()
        groups.setdefault(key,[]).append((m[2],len(rationale.split())))
        order.append(key)
    clean=bool(matches) and len(strict)==len(IDS) and order==list(map(str,IDS)) and not text[:matches[0].start()].strip()
    items={}
    for i in map(str,IDS):
        entries=groups.get(i,[])
        items[i]={'occurrences':len(entries),'correct':len(entries)==1 and entries[0][0]==gold[i],
                  'rationale_words':entries[0][1] if len(entries)==1 else None}
    words=re.findall(r"\w+",text.casefold())
    windows=[tuple(words[j:j+24]) for j in range(max(0,len(words)-23))]
    counts=Counter(windows)
    book_words=re.findall(r"\w+",(PRIVATE/'book.txt').read_text().casefold())
    book=set(tuple(book_words[j:j+24]) for j in range(len(book_words)-23))
    lines=[' '.join(line.casefold().split()) for line in text.splitlines() if line.strip()]
    total=sum(int(v['correct']) for v in items.values())
    return {'choice_correct':total,'choice_total':4,'items':items,'all_choices_correct':total==4,
            'malformed':not clean or any(not v['rationale_words'] for v in items.values()),
            'blind_rationale_grounding':[None]*4,'nonblank_lines':len(lines),
            'duplicate_lines':len(lines)-len(set(lines)),'word_windows_24':len(windows),
            'repeated_word_windows_24':len(windows)-len(counts),
            'book_copied_word_windows_24':sum(n for w,n in counts.items() if w in book)}


def result(phase,out):
    d=read(out/f'{phase}.json')
    need(d['phase']==phase and d['settings']==expected and d['prompt_sha256']==fixture['prompt_ids_sha256'],'request fixture mismatch')
    f=d['terminal']; t=f['timings']; conf=f['generation_settings']
    need(f['stop'] is True and f['tokens_evaluated']==80000 and f['id_slot']==0,'missing terminal identity')
    need(all(conf[k]==expected[k] for k in ('temperature','seed','ignore_eos','n_predict','stop')),'actual sampler changed')
    need(all(type(t[k]) is int for k in ('predicted_n','cache_n','prompt_n')) and
         t['cache_n']+t['prompt_n']==80000 and t['prompt_n']>0 and (t['cache_n']==0 if phase=='cold' else t['cache_n']>0),'incorrect prompt/cache accounting')
    n=t['predicted_n']; ms=t['predicted_ms']
    need(1<=n<=1536 and n==f['tokens_predicted'] and ms>0 and math.isfinite(ms),'bad decode counts or clock')
    rate=(n-1)*1000/ms
    need(math.isclose(rate,t['predicted_per_second'],rel_tol=2e-6,abs_tol=0.02),'server count/clock/rate mismatch')
    with (out/f'{phase}-events.jsonl').open() as src: events=[json.loads(x) for x in src]
    need(events[-1]['terminal']==f and sum(x['terminal'] is not None for x in events)==1 and
         ''.join(x['text'] for x in events)==d['text'] and [i for x in events for i in x['tokens']]==d['stream_ids'],'SSE replay mismatch')
    times=[x['elapsed_ns'] for x in events]
    need(all(type(v) is int and v>=0 for v in times) and times==sorted(times) and times[-1]<=d['wall_ns'],'bad client clock')
    token_events=[x for x in events if x['tokens']]
    emitted=[x for x in events if x['text']]
    stream_n=len(d['stream_ids'])
    need(0<stream_n<=n and token_events,'bad streamed ID count')
    vis=len(d['visible_ids'])
    obs=score(d['text'])
    word=f['stop_type']=='word' and f['stopping_word']==STOP and f['truncated'] is False
    raw_suffix=d['raw_decoded'].find(STOP)
    reconstructed_word=(raw_suffix>=0 and d['raw_decoded'][:raw_suffix]==d['text'])
    first=token_events[0]; last=token_events[-1]; span=(last['elapsed_ns']-first['elapsed_ns'])/1e9
    elapsed=d['wall_ns']/1e9
    visible_span=(emitted[-1]['elapsed_ns']-emitted[0]['elapsed_ns'])/1e9 if emitted else 0
    return {'phase':phase,'prompt_ids_sha256':fixture['prompt_ids_sha256'],'cache_n':t['cache_n'],'evaluated_n':t['prompt_n'],
            'prompt_ms':t['prompt_ms'],'raw_generated_n':n,'server_decode_steps':n-1,'server_decode_ms':ms,'server_decode_tps':rate,
            'visible_retokenized_n':vis,'sustained':vis>=512,'streamed_id_n':stream_n,'all_generated_ids_streamed':stream_n==n,
            'token_ttft_s':first['elapsed_ns']/1e9,'visible_ttft_s':emitted[0]['elapsed_ns']/1e9 if emitted else None,
            'first_last_stream_s':span,'stream_id_tps':(stream_n-len(first['tokens']))/span if span else None,
            'first_last_visible_event_s':visible_span,'request_wall_s':elapsed,'visible_end_to_end_tps':vis/elapsed,
            'stop_type':f['stop_type'],'truncated':f['truncated'],'declared_word_observed':word,
            'word_verified_in_raw_ids':reconstructed_word,'task_terminal':word and (reconstructed_word if stream_n==n else True),
            'offered':t.get('draft_n',0),'accepted':t.get('draft_n_accepted',0),
            'stream_ids_sha256':digest(packed(d['stream_ids'])),'visible_text_sha256':digest(d['text'].encode()),'quality':obs}


def witness(mode,out,phases,metrics):
    logs=[(out/'server-start.log').read_text(),(out/'server-pre-warm.log').read_text()]
    if 'warm' in phases: logs.append((out/'server.log').read_text())
    final=(out/'server.log').read_text()
    need(all(b.startswith(a) for a,b in zip(logs,logs[1:])) and final.startswith(logs[-1]),'noncumulative server logs')
    need(not r2.q2.n88.short.CUDA_ERROR.search(final) and 'context shift' not in final.lower().replace('context shift is disabled',''),'device error or context shift')
    if mode=='native':
        r2.hybrid.graph_witness(out)
    else: need(r2.hybrid.MERGE not in final and 'mtp_last_graph' not in final,'unexpected serial draft graph')
    captures=[read(out/'witness'/f'{p}.json') for p in ['start',*phases]]
    pid=int((out/'witness/child.pid').read_text())
    need(all(c['child']==pid and all(type(v) is int and v>=0 for v in c.values()) for c in captures),'bad physical witness')
    residency=r'^Q2_DOWN4 residency device=0 sm=24 active=2 regs=62 static=96 dynamic=39424 local=0 ready=1$'
    need(len(re.findall(residency,final,re.M))==final.count('Q2_DOWN4 residency')==int(captures[-1]['down4']>0),'down4 residency mismatch')
    for n,ph in enumerate(phases):
        part=logs[n+1][len(logs[n]):]
        m=metrics[ph]; offers=m['offered']; accepted=m['accepted']
        need(type(offers) is int and type(accepted) is int and 0<=accepted<=offers,'bad offer/accepted units')
        need((offers>0 if mode=='native' and m['raw_generated_n']>3 else offers==0 if mode=='serial' else True),'mode offers disagree')
        m['accepted_fraction']=accepted/offers if offers else None
        delta={k:v-captures[n][k] for k,v in captures[n+1].items() if k!='child'}
        need(min(delta.values())>=0 and delta['row0']==delta['row1']==0 and
             delta['down4']==delta['combined']==delta['skip']==offers*40,'down4 launch attribution')
        for pattern,keys,tag,marker in [(r2.MARKER,r2.COUNTERS,'compact','compact_readback:'),(r2.GUARD_MARKER,r2.GUARDS,'guard','compact_guard:')]:
            rows=pattern.findall(part)
            need(len(rows)==1 and part.count(marker)==1,'missing or duplicate compact terminal witness')
            prev=pattern.findall(logs[n])
            before=map(int,prev[-1]) if prev else [0]*len(keys)
            m[tag]=dict(zip(keys,[int(b)-a for a,b in zip(before,rows[0])]))
            need(min(m[tag].values())>=0,'counter regressed')
        g,c=m['guard'],m['compact']
        need(sum(g.values())==offers and c['requested']==g['ready'] and c['launched']<=c['requested'] and
             c['compact_only']<=c['launched'] and c['compact_bytes']==16*c['launched'] and
             c['shadow_rows']==c['shadow_mismatches']==0,'guard/compact units disagree')
        need(c['raw_bytes']==1986560*(c['requested']-c['compact_only']) and
             c['compact_only']+c['ambiguous']+c['unsupported']==c['requested'],'raw readback units disagree')
        m['down4_physical_delta']=delta
        m['all_offers_compact_only']=(c['compact_only']==offers if offers else None)
    return metrics


def collect(mode,out):
    _,mapped=r2.ready(mode,out)
    phases=['cold']+(['warm'] if (out/'warm.json').exists() else [])
    metrics={ph:result(ph,out) for ph in phases}
    return {'mode':mode,'experts_used':8,'fixture_sha256':digest((CARD/'fixture.json').read_bytes()),
            'mapped':mapped,'phases':witness(mode,out,phases,metrics)}


def checked(mode,p):
    p=p.resolve()
    need(p.parent==CARD/'receipts' and p.name.startswith(mode+'-'),'receipt owner or mode mismatch')
    subprocess.run(['sha256sum','-c','--quiet','observations.sha256'],cwd=p,check=True)
    status=dict(x.split('=',1) for x in (p/'status.txt').read_text().splitlines())
    m=(p/'manifest.txt').read_text().splitlines()
    need(status['runner_exit']==status['cleanup_exit']=='0' and
         f'{digest((CARD/"FROZEN.sha256").read_bytes())}  {CARD}/FROZEN.sha256' in m,'nonterminal or different freeze')
    x=collect(mode,p)
    need(x==read(p/'result.json')==read(p/'terminal.json')['observed'],'terminal replay changed')
    return x,int(next(k[12:] for k in m if k.startswith('start_epoch='))),int(status['completed_epoch'])


def compare(paths):
    modes=['serial','native','native','serial']
    arms=[checked(mode,p) for mode,p in zip(modes,paths,strict=True)]
    need(len({p.resolve() for p in paths})==4 and all(a[2]<=b[1] for a,b in zip(arms,arms[1:])),'require independent completed ABBA order')
    pairs=[]
    for s,n in [(0,1),(3,2)]:
        a,b=arms[s][0],arms[n][0]
        need(a['mapped']==b['mapped'] and a['fixture_sha256']==b['fixture_sha256'],'fixture or DSO mismatch')
        ratio={}
        for ph in a['phases'].keys() & b['phases'].keys():
            sm,nm=a['phases'][ph],b['phases'][ph]
            same_cache=(sm['cache_n']==nm['cache_n'])
            ratio[ph]={'same_cache':same_cache,'native_over_serial_server_rate':nm['server_decode_tps']/sm['server_decode_tps'] if same_cache and sm['server_decode_tps'] else None,
                       'same_visible_text':sm['visible_text_sha256']==nm['visible_text_sha256']}
        pairs.append(ratio)
    return {'stage_order':modes,'arms':[x[0] for x in arms],'paired_observations':pairs}


def main():
    cmd,*args=sys.argv[1:]
    if cmd=='check':
        need(len(read(PRIVATE/'ids.json'))==80000,'invalid prompt length')
        for k,h in fixture['private_sha256'].items():
            need(digest((PRIVATE/k).read_bytes())==h and ((PRIVATE/k).stat().st_mode & 0o077)==0,'private fixture mismatch')
        need(r2.settings('serial')[0]==r2.settings('native')[0],'different typed flags')
        need(score('')['malformed'] and score('')['choice_correct']==0,'malformed score')
    elif cmd in ('env','args'):
        print('\n'.join(r2.settings(args[0])[cmd=='args']))
    elif cmd=='ready': r2.ready(args[0],Path(args[1]))
    elif cmd=='verify': write(Path(args[1])/'result.json',collect(args[0],Path(args[1])))
    elif cmd=='probe': print(json.dumps(collect(args[0],Path(args[1])),indent=2))
    elif cmd=='finalize':
        mode,folder,rc,cleanup,elapsed=args
        out=Path(folder)
        observed=read(out/'result.json') if (out/'result.json').exists() else None
        write(out/'terminal.json',{'mode':mode,'runner_exit':int(rc),'cleanup_exit':int(cleanup),
             'arm_wall_seconds':int(elapsed),'observed':observed,'private_event_logs':[p.name for p in out.glob('*-events.jsonl')]})
    elif cmd=='compare': print(json.dumps(compare(list(map(Path,args))),indent=2,allow_nan=False))
    else: raise ValueError('unknown subcommand')


if __name__=='__main__':
    main()
