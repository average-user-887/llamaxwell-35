"""Build/check one completely local book-reading adaptation. No model inference."""
import hashlib
import json
import os
from pathlib import Path
import string
import sys

CARD = Path(__file__).resolve().parent
PRIVATE = CARD / 'private'
SOURCE = (Path.home() / '.cache/huggingface/hub/datasets--xinrongzhang2022--InfiniteBench/snapshots'
          / '90f0394333616266d9fe85824ceaf505093cbaa5/longbook_choice_eng.jsonl')
MODEL = CARD.parents[2] / 'uroboros-models/occamy-1.0-Q4_0-headq40b-mtp.gguf'
MODEL_HASH = 'd5ff985eea0bcb3a7305a2033582bf63f756c9f4b3700d96b4939f81ee041eae'
STOP = '\nEND OF READING REPORT'
IDS = [150, 151, 152, 153]
SHA = 'd2e646822a3d40dc329e57894d81cf9c527503e963ba06567c3eaaccf78f87fe'


def need(b, msg):
    if not b:
        raise ValueError(msg)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def packed(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def read(p):
    return json.loads(p.read_bytes())


def data():
    out = {}
    h = hashlib.sha256()
    with SOURCE.open('rb') as f:
        for i, line in enumerate(f):
            h.update(line)
            if i in (0, 108, 109, 110, 111):
                row = json.loads(line)
                out[row['id']] = row
    need(h.hexdigest() == SHA and set(out) == {0, *IDS}, 'dataset changed')
    return out


def tokenizer(r):
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers
    names = r.get_field('tokenizer.ggml.tokens').contents()
    types = r.get_field('tokenizer.ggml.token_type').contents()
    merges = [tuple(x.split()) for x in r.get_field('tokenizer.ggml.merges').contents()]
    need(r.get_field('tokenizer.ggml.pre').contents() == 'qwen2', 'unexpected pretokenizer')
    t = Tokenizer(models.BPE(vocab={v:i for i,v in enumerate(names)}, merges=merges))
    rx = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(rx),'isolated'),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
    t.add_special_tokens([AddedToken(v,normalized=False,special=True) for v,k in zip(names,types) if k in (3,4)])
    return t


def freeze(write):
    from gguf import GGUFReader
    from jinja2 import Environment
    rows = data()
    book = rows[150]['context']
    need(all(rows[i]['context'] == book for i in IDS), 'book mismatch')
    need(digest(book.encode()) == 'd3fa1ef5e2ffc10becdfc8114abdecef784a6f354ea62d987b39065d7bb36b50','book digest')
    gold = {}
    qs = []
    for i in IDS:
        x = rows[i]
        need(len(x['options'])==4 and len(x['answer'])==1 and x['options'].count(x['answer'][0])==1,'choice format')
        gold[str(i)] = string.ascii_uppercase[x['options'].index(x['answer'][0])]
        qs.append(str(i)+'. '+x['input']+'\n'+'\n'.join(f'{c}. {a}' for c,a in zip('ABCD', x['options'])))
    r = GGUFReader(MODEL)
    t = tokenizer(r)
    tmpl = r.get_field('tokenizer.chat_template').contents()
    template = Environment(trim_blocks=True,lstrip_blocks=True).from_string(tmpl)
    before = 'Reading packet. The opening appendix is from a different work and is included for reference. Answer the questions about the complete book that follows.\n\nRELATED READING APPENDIX (separate work)\n'
    after = '\n\nEND APPENDIX\n\nCOMPLETE BOOK\n'+book+'\n\nEND BOOK\n\nREADING QUESTIONS\n'+'\n\n'.join(qs)
    after += ('\n\nFor each question in order write one paragraph in the form “150 A. Rationale” using the actual question number and chosen letter. '
              'Give roughly 125 words per paragraph. Explain the choice with a distinct supporting detail and why a plausible alternative fails. '
              'Use your own words. Finish after the fourth paragraph with this exact line: END OF READING REPORT')
    def rendered(msg):
        return template.render(messages=[dict(role='user',content=msg)],add_generation_prompt=True,enable_thinking=False)
    # An independent owned-server archive proves this renderer and tokenizer agree on complete
    # chat boundaries, controls and actual non-placeholder token spans.
    c = CARD.parent
    n72=c/'night-lookup-heldout'
    archived=c/'night-lookup-heldout-r1/receipts/night-n72-r1-bounded-20261118T045733Z-3100349/private'
    old_msg=(n72/'TASK.txt').read_text()+(n72/'donor.before.py').read_text()+'</OLD_FILE>'
    ref=read(archived/'rendered-template.json')
    need(rendered(old_msg)==ref,'offline renderer differs from archived owned template')
    front,back=ref.split('N72_FROZEN_REFERENCE_CONTEXT_65A296')
    known=read(archived/'prompt-token-ids.json')
    pre,tail=t.encode(front).ids,t.encode(back).ids
    need(known[:len(pre)]==pre and known[-len(tail):]==tail,'offline tokenizer disagrees with owned prompt boundaries')
    appendix = rows[0]['context'][:80000]
    # Measure full tokenization near the estimate. Only the appendix prefix may be adjusted.
    needed = 80000-len(t.encode(rendered(before+after)).ids)
    a = t.encode(appendix)
    approx = a.offsets[needed-1][1]
    for delta in [0]+[y for x in range(1,160) for y in (x,-x)]:
        chars=approx+delta
        msg=before+appendix[:chars]+after
        render=rendered(msg)
        ids=t.encode(render).ids
        if len(ids)==80000:
            break
    need(len(ids)==80000,'full rendered text did not give exact 80000 tokens')
    items = {'message.txt':msg.encode(),'rendered.txt':render.encode(),'ids.json':packed(ids),
             'gold.json':packed(gold),'book.txt':book.encode()}
    manifest = {'adaptation':'InfiniteBench local four-question reading report','source_sha256':SHA,'ids':IDS,
                'selected_book_sha256':digest(book.encode()),'appendix_row':0,'appendix_chars':chars,
                'book_tokens':len(t.encode(book).ids),'prompt_tokens':len(ids),'prompt_ids_sha256':digest(packed(ids)),
                'template_sha256':digest(tmpl.encode()),'rendered_sha256':digest(render.encode()),
                'private_sha256':{k:digest(v) for k,v in items.items()},'seed':7,'temperature':0,
                'cap':1536,'sustained_visible_minimum':512,'stop':STOP,'model_sha256':MODEL_HASH}
    for name,raw in [*( (PRIVATE/k,v) for k,v in items.items()), (CARD/'fixture.json',packed(manifest))]:
        if write:
            with name.open('xb') as f: f.write(raw)
        else:
            need(name.read_bytes()==raw,'freeze replay differs for '+name.name)
    print('Verified exact 80000-token book fixture and four unique gold option mappings.')


if __name__=='__main__':
    os.umask(0o077)
    need(sys.argv[1:] in (['--freeze'],['--check']),'use --freeze or --check')
    freeze(sys.argv[1]=='--freeze')
