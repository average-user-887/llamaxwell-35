"""Compare private replay files and emit anonymous aggregate differences."""
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB = 248320
PARTIAL_BYTES = 65864428
RECURRENT = tuple(i for i in range(40) if (i+1) % 4)
STRIDES = (('conv', 98304), ('ssm', 2097152))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def delta(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype==np.dtype('<f4'),'incommensurate F32 rows')
    bits_a,bits_b=a.view('<u4'),b.view('<u4')
    changed=bits_a!=bits_b
    count=int(np.count_nonzero(changed))
    res={'f32_count':a.size,'different_words':count,'different_bytes':int(np.count_nonzero(a.view('u1')!=b.view('u1'))),
         'nonfinite_a':a.size-int(np.count_nonzero(np.isfinite(a))),'nonfinite_b':b.size-int(np.count_nonzero(np.isfinite(b)))}
    if res['nonfinite_a'] or res['nonfinite_b']:
        return {**res,'max_abs':None,'rms':None}
    if not count:
        return {**res,'max_abs':0.0,'rms':0.0}
    diff=np.abs(np.subtract(a,b,dtype=np.float64))
    j=int(diff.argmax())
    return {**res,'max_abs':float(diff[j]),'rms':float(np.sqrt(np.mean(diff*diff)))}


class State:
    def __init__(self,path,position):
        require(path.stat().st_size==PARTIAL_BYTES,'wrong partial-state length')
        self.buf=np.memmap(path,mode='r',dtype='u1')
        # Serialized sequence API writes its magic and sequence ID, then hybrid partial writes recurrent only.
        header=struct.unpack_from('<IIIiIII',self.buf)
        require(header==(0xaf143cd8,0,1,position,0,0,40),'unexpected recurrent metadata')
        offset=28
        self.metadata=bytearray(self.buf[:offset])
        self.payload={}
        for kind,stride in STRIDES:
            for layer in RECURRENT:
                require(struct.unpack_from('<iQ',self.buf,offset)==(0,stride),'unexpected recurrent type/row size')
                self.metadata.extend(self.buf[offset:offset+12])
                offset+=12
                self.payload[kind,layer]=np.frombuffer(self.buf,dtype='<f4',count=stride//4,offset=offset)
                offset+=stride
        require(offset==len(self.buf),'partial-state payload not fully consumed')


def compare_state(a,b):
    require(a.payload.keys()==b.payload.keys(),'state shapes differ')
    by_layer=[]
    for (kind,layer),x in a.payload.items():
        by_layer.append({'layer':layer,'kind':kind,**delta(x,b.payload[kind,layer])})
    count=sum(x['different_words'] for x in by_layer)
    nonfinite=sum(x['nonfinite_a']+x['nonfinite_b'] for x in by_layer)
    return {'raw_bitwise_equal':a.metadata==b.metadata and count==0,'metadata_bytes':len(a.metadata),
            'metadata_equal':a.metadata==b.metadata,'payload_f32_words':sum(x['f32_count'] for x in by_layer),
            'different_f32_words':count,'different_payload_bytes':sum(x['different_bytes'] for x in by_layer),
            'nonfinite_values_a_plus_b':nonfinite,'max_abs':None if nonfinite else max(x['max_abs'] for x in by_layer),
            'nonidentical_layer_rows':sum(x['different_words']>0 for x in by_layer),'per_layer':by_layer}


def logits(path,rows):
    require(path.stat().st_size==rows*VOCAB*4,'wrong logit row length')
    return np.memmap(path,'<f4',mode='r',shape=(rows,VOCAB))


def fingerprint(directory, name):
    private = directory/name
    if private.is_file():
        with private.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").digest()
    return bytes.fromhex((directory/(name+".sha256")).read_text().split()[0])

def compare(directory):
    results = {}
    for kind in ("state", "kv"):
        prefix = {fingerprint(directory,f"{arm}.prefix.{kind}") for arm in ("wide_a","wide_b","serial")}
        require(len(prefix)==1, "independently prefetched initial states differ")
        require(fingerprint(directory,"wide_a.badtriple."+kind)!=fingerprint(directory,"wide_b.badtriple."+kind),
                "different rejected suffixes failed to change the untrimmed state")
    for b in ("wide_b","serial"):
        a="wide_a"
        result={"logit_row_deltas":{}}
        for stage,pos in (("kept",2004),("corrected",2005)):
            recurrent=compare_state(State(directory/f"{a}.{stage}.state",pos),State(directory/f"{b}.{stage}.state",pos))
            x,y=[np.memmap(directory/f"{arm}.{stage}.kv","u1","r") for arm in (a,b)]
            require(x.shape==y.shape,"attention blob size mismatch")
            result[stage]={"recurrent":recurrent,"attention_kv":{"bytes":int(x.size),"different_bytes":int(np.count_nonzero(x!=y))}}
            def row(arm):
                wide_kept = stage=="kept" and arm!="serial"
                name="badtriple" if wide_kept else stage
                return logits(directory/f"{arm}.{name}.logits",3 if wide_kept else 1)[0]
            result["logit_row_deltas"][stage]=delta(row(a),row(b))
        def choices(arm):
            text=(directory/f"{arm}.rows").read_text()
            picked,=re.findall(r"^N34_END .*?first_choice=(\d+)$",text,re.M)
            follow=re.findall(r"^N34_CALL stage=(?:corrected|follow) .*?next0=(\d+)",text,re.M)
            require(len(follow)==5,"missing corrected or following choices")
            return picked,follow
        i,x=choices(a)
        j,y=choices(b)
        result.update(first_choice_equal=i==j,choices_equal=x==y,next_choices_equal_after_correction_and_four_replay_steps=[u==v for u,v in zip(x,y)])
        results[a+"_vs_"+b]=result
    return results

if __name__=="__main__":
    results=compare(Path(sys.argv[1]))
    if len(sys.argv)>2:
        require(results==json.loads(Path(sys.argv[2]).read_text())["comparisons"],"public aggregate differs from private rerun")
        print("All anonymous comparisons reproduced.")
    else:
        print(json.dumps(results,indent=2,allow_nan=False))
