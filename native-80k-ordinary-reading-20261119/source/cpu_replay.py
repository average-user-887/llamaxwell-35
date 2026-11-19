"""Replay terminal paths and an archived R2 native physical trace without a server."""
import tempfile
import shutil
from pathlib import Path

from fixture import CARD, IDS, PRIVATE, STOP, packed, read, need
import model

def main():
    old=CARD.parent/'break100-compact-model-r2/receipts/native-20261119T124013Z-3647221-4956'
    need(old.is_dir(),'pinned CPU witness missing')
    model.r2.ready('native',old)
    old_t=read(old/'cold.json')['timings']
    with tempfile.TemporaryDirectory(dir=CARD) as tmp:
        out=Path(tmp)
        (out/'witness').mkdir()
        for name in ('server-start.log','server-pre-warm.log'):
            shutil.copy(old/name,out/name)
        shutil.copy(old/'server-pre-warm.log',out/'server.log')
        for name in ('start.json','cold.json','child.pid'):
            shutil.copy(old/'witness'/name,out/'witness'/name)
        m={'offered':old_t['draft_n'],'accepted':old_t['draft_n_accepted'],'raw_generated_n':old_t['predicted_n']}
        w=model.witness('native',out,['cold'],{'cold':m})['cold']
        need(w['all_offers_compact_only'] and w['down4_physical_delta']['down4']==40*w['offered'],'cold witness replay')
        text='\n'.join(f'{i} {read(PRIVATE/"gold.json")[str(i)]}. An original explanation.' for i in IDS)
        for kind in ('word','limit','eos','wrong_word'):
            expected_word=kind=='word'
            body=text if expected_word else 'malformed answer'
            final={'id_slot':0,'tokens_evaluated':80000,'tokens_predicted':3,'stop':True,
                   'stop_type':kind if kind!='wrong_word' else 'word','stopping_word':STOP if kind=='word' else '',
                   'generation_settings':model.expected,'truncated':False,
                   'timings':{'predicted_n':3,'predicted_ms':250,'predicted_per_second':8,
                              'cache_n':0,'prompt_n':80000,'prompt_ms':2000}}
            events=[{'elapsed_ns':x*1000000,'tokens':[x],'text':body if x==1 else '','terminal':None} for x in [1,2,3]]
            events.append({'elapsed_ns':4000000,'tokens':[],'text':'','terminal':final})
            (out/'cold-events.jsonl').write_bytes(b'\n'.join(map(packed,events)))
            model.write(out/'cold.json',dict(phase='cold',settings=model.expected,prompt_sha256=model.fixture['prompt_ids_sha256'],
                terminal=final,text=body,raw_decoded=body+(STOP if expected_word else ''),visible_ids=[1],stream_ids=[1,2,3],wall_ns=5000000))
            d=model.result('cold',out)
            need(d['task_terminal']==expected_word and d['quality']['all_choices_correct']==expected_word and not d['sustained'],'terminal score replay')
    print('CPU replay passed for the physical cold trace, four option keys, malformed, early, capped and wrong stops.')


if __name__=='__main__': main()
