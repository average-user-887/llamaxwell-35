"""Acquire one private response via the already admitted owned loopback server."""
import json
from pathlib import Path
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request

from fixture import CARD, PRIVATE, STOP, digest, need, packed, read


def run(phase, port, out):
    need(phase in ('cold','warm') and 0 < port < 65536, 'unexpected phase/port')
    opener = build_opener(ProxyHandler({}))
    def req(route, body, secs):
        addr = f'http://127.0.0.1:{port}/{route}'
        return opener.open(Request(addr,packed(body),{'Content-Type':'application/json'}), timeout=secs)
    def post(route, body):
        with req(route,body,25) as response:
            return json.load(response)
    def tokenize(text, special=False):
        return post('tokenize',{'content':text,'add_special':False,'parse_special':special})['tokens']
    ids = read(PRIVATE/'ids.json')
    msg = (PRIVATE/'message.txt').read_text()
    if phase == 'cold':
        rendered = post('apply-template',{'messages':[{'role':'user','content':msg}],
                                         'chat_template_kwargs':{'enable_thinking':False}})['prompt']
        need(rendered==(PRIVATE/'rendered.txt').read_text(), 'owned template differs')
        need(tokenize(rendered,True)==ids, 'owned prompt tokenization differs')
    settings = {'temperature':0,'seed':7,'n_predict':1536,'ignore_eos':True,'stop':[STOP],
                'id_slot':0,'cache_prompt':True,'stream':True,'return_tokens':True}
    start = time.monotonic_ns()
    events = out/f'{phase}-events.jsonl'
    parts=[]; sampled=[]; ending=None
    with events.open('x',buffering=1) as record, req('completion',dict(prompt=ids,**settings),375) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            ns=time.monotonic_ns()-start
            p=json.loads(line[5:])
            need(isinstance(p,dict) and 'error' not in p and ending is None,'malformed or late SSE event')
            chunk=p.get('tokens',[]); txt=p.get('content','')
            need(type(chunk)==list and all(type(t)==int and t>=0 for t in chunk) and type(txt)==str,'bad SSE payload')
            if p.get('stop') is True:
                ending=p
            parts.append(txt); sampled.extend(chunk)
            record.write(packed({'elapsed_ns':ns,'tokens':chunk,'text':txt,'terminal':p if ending else None}).decode()+'\n')
            if ending: break
    wall=time.monotonic_ns()-start
    need(ending is not None,'missing terminal SSE event')
    text=''.join(parts)
    visible=tokenize(text)
    decoded=post('detokenize',{'tokens':sampled})['content']
    (out/f'{phase}.json').write_bytes(packed({'phase':phase,'settings':settings,'prompt_sha256':digest(packed(ids)),
        'text':text,'visible_ids':visible,'stream_ids':sampled,'raw_decoded':decoded,'terminal':ending,'wall_ns':wall}))


if __name__=='__main__':
    phase,port,out=sys.argv[1:]
    run(phase,int(port),Path(out))
