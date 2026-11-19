"""Exercise all three validators on disposable synthetic copies of the previous pair."""
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import model


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def need_refusal(fn):
    try:
        fn()
    except ValueError:
        return
    raise ValueError('counterexample passed')


def counters(stage, offers):
    shadow=stage=='shadow'
    ambiguous=int(offers>0)
    ok=offers-ambiguous
    compact=0 if shadow else ok
    values=(offers,offers,compact,ambiguous,0,1986560*(offers-compact),16*offers,2*ok if shadow else 0,0)
    guard='compact_guard: '+' '.join(f'{k}={offers if k=="ready" else 0}' for k in model.GUARDS)+'\n'
    return guard + 'compact_readback: '+' '.join(f'{k}={v}' for k,v in zip(model.COUNTERS,values))+'\n'


def checksum(folder):
    names=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='observations.sha256')
    (folder/'observations.sha256').write_text(''.join(model.q2.n9.hash_file(f)+'  ./'+str(f.relative_to(folder))+'\n' for f in names))


def main():
    with tempfile.TemporaryDirectory(dir=model.HERE) as t:
        temp=Path(t)
        (temp/'receipts').mkdir()
        shutil.copy2(model.HERE/'supervise',temp/'supervise')
        (temp/'FROZEN.sha256').write_text('SYNTHETIC ONLY\n')
        with patch.object(model,'HERE',temp):
            runs={}
            for stage in model.STAGES:
                underlying=model.mode(stage)
                src=model.ARCHIVE[underlying]
                out=temp/'receipts'/(stage+'-synthetic')
                shutil.copytree(src,out)
                runs[stage]=out
                env,_=model.settings(stage)
                (out/'environment.args').write_text('\n'.join(env)+'\n')
                with (out/'child-selected-environment.txt').open('a') as f:
                    f.write(env[-3]+'\n'+env[-1]+'\n')
                f=out/'runtime-library-hashes.txt'
                inventory=f.read_text()
                base_digests=model.read(src/'result.json')['actual_server_software_sha256']
                for name,v in model.profile.items():
                    if v != base_digests[name]:
                        assert inventory.count(base_digests[name])==1
                        inventory=inventory.replace(base_digests[name],v)
                f.write_text(inventory)
                (out/'supervisor.sha256').write_text(model.q2.n9.hash_file(temp/'supervise')+'  /proc/1/exe\n')
                for phase in ('start','cold','warm'):
                    f=out/'witness'/f'{phase}.json'
                    data=model.read(f)
                    data['down4']=0
                    save(f,data)
                n=int(model.read(out/'cold.json')['timings'].get('draft_n') or 0)
                warm=int(model.read(out/'warm.json')['timings'].get('draft_n') or 0)
                pre, full = out/'server-pre-warm.log', out/'server.log'
                prefix, complete = pre.read_text(), full.read_text()
                assert complete.startswith(prefix)
                added = counters(stage,n)
                pre.write_text(prefix+added)
                full.write_text(prefix+added+complete[len(prefix):]+counters(stage,n+warm))
                f=out/'manifest.txt'
                old=f.read_text().splitlines()
                f.write_text('stage='+stage+'\n'+'\n'.join(old[1:])+'\n'+model.q2.n9.hash_file(temp/'FROZEN.sha256')+'  '+str(temp/'FROZEN.sha256')+'\n')
                record=model.collect(stage,out)
                save(out/'result.json',record)
                f=out/'sanitized/model-receipt.json'
                official=model.read(f)
                official.update(mode=stage,validated=record)
                save(f,official)
                checksum(out)
                assert model.finished(stage,out)==record
            pair=model.compare(runs['serial'],runs['native'])
            assert pair['same_build']==model.profile and pair['phases']['warm']['observed_100_tps'] is False
            native=runs['native']
            f=native/'server.log'; original=f.read_text()
            for old,new in [('launched=1743','launched=1742'), ('raw_bytes=1986560','raw_bytes=0'), ('compact_bytes=27888','compact_bytes=1'), ('shadow_mismatches=0','shadow_mismatches=1'), ('types=0','types=1')]:
                assert old in original
                f.write_text(original.replace(old,new))
                need_refusal(lambda:model.collect('native',native))
            f.write_text(original)
            f=native/'runtime-library-hashes.txt'
            f.write_text(f.read_text().replace(model.profile['libllama.so.0.6.0'],'0'*64))
            need_refusal(lambda:model.ready('native',native))
    print('SYNTHETIC ONLY native/serial/shadow complete validator replay and six negative controls passed')


if __name__=='__main__':
    main()
