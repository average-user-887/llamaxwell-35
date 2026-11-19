"""Replay positive/negative checks on temporary copies of the recorded two-arm requests."""
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import model

def need_refusal(fn):
    try: fn()
    except ValueError: return
    raise ValueError('counterexample passed')

def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def checksum(folder):
    names=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='observations.sha256')
    (folder/'observations.sha256').write_text(''.join(model.q2.n9.hash_file(f)+'  ./'+str(f.relative_to(folder))+'\n' for f in names))

def main():
    with tempfile.TemporaryDirectory(dir=model.HERE) as t:
        testroot=Path(t)
        (testroot/'receipts').mkdir()
        shutil.copy2(model.HERE/'supervise',testroot/'supervise')
        (testroot/'FROZEN.sha256').write_text('CPU-SYNTHETIC-ONLY\n')
        with patch.object(model,'HERE',testroot):
            runs={}
            for mode,src in model.ARCHIVE.items():
                out=testroot/'receipts'/(mode+'-synthetic')
                shutil.copytree(src,out);runs[mode]=out
                env,_=model.settings(mode)
                (out/'environment.args').write_text('\n'.join(env)+'\n')
                with (out/'child-selected-environment.txt').open('a') as f:f.write(model.flag+'\n')
                f=out/'runtime-library-hashes.txt'
                f.write_text(f.read_text().replace('369c6b058813a7ca53fc62c67c2a4ea056a8d3e35ee08aca36afca4040975871',model.profile['libggml-cuda.so']))
                (out/'supervisor.sha256').write_text(model.q2.n9.hash_file(testroot/'supervise')+'  /proc/1/exe\n')
                for phase in ('start','cold','warm'):
                    f=out/'witness'/f'{phase}.json';data=model.read(f);data['down4']=data['combined'];save(f,data)
                marker='Q2_DOWN4 residency device=0 sm=24 active=2 regs=62 static=96 dynamic=39424 local=0 ready=1'
                for n in ('server-start.log','server-pre-warm.log','server.log'):
                    f=out/n; f.write_text(marker+'\n'+f.read_text())
                with (out/'manifest.txt').open('a') as f:f.write(model.q2.n9.hash_file(testroot/'FROZEN.sha256')+'  '+str(testroot/'FROZEN.sha256')+'\n')
                record=model.collect(mode,out);save(out/'result.json',record)
                for n in ['sanitized/model-receipt.json']:
                    f=out/n;official=model.read(f);official['validated']=record;save(f,official)
                checksum(out)
                result=model.finished(mode,out)
                print(mode,result['down4'])
            pair=model.compare(runs['serial'],runs['native'])
            assert pair['same_build']==model.profile and pair['phases']['warm']['observed_100_tps'] is False
            native=runs['native']
            f=native/'witness/warm.json';original=model.read(f)
            bad=dict(original,down4=original['down4']-1);save(f,bad)
            need_refusal(lambda:model.collect('native',native));save(f,original)
            serial=runs['serial'];f=serial/'witness/start.json';original=model.read(f)
            save(f,dict(original,down4=original['down4']-1))
            need_refusal(lambda:model.collect('serial',serial));save(f,original)
            f=native/'runtime-library-hashes.txt';f.write_text(f.read_text().replace(model.profile['libggml-cuda.so'],'0'*64))
            need_refusal(lambda:model.ready('native',native))
            print('CPU_SYNTHETIC_ONLY both complete receipts and strict pair replayed; one missing candidate, serial leak and wrong mapped DSO refused')

if __name__=='__main__':main()
