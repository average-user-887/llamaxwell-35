"""Real historical request/log replay with explicitly synthetic counter and new runtime-map fields."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,os
import model as p
def write(path,data):path.write_text(json.dumps(data)+'\n')
def stage(root,mode,ref):
    o=root/mode;o.mkdir();w=o/'witness';w.mkdir()
    for name in ('admission.txt','cold.json','warm.json','server-start.log','server-pre-warm.log','server.log'):
        if name=='server.log' and mode=='native':
            (o/name).write_text((ref/name).read_text()+'\nQ2_COHORT residency device=0 sm=24 active=2 regs=63 static=96 dynamic=39424 local=0 ready=1\n')
        else:(o/name).symlink_to(ref/name)
    env,args=p.settings(mode)
    for name,data in [('environment.args',env),('server.args',args),('child-selected-environment.txt',env[1::2]),('child-server.args',['/candidate/llama-server',*args])]:
        (o/name).write_text('\n'.join(data)+'\n')
    (w/'child.pid').write_text('99\n')
    soft=p.base.maps(p.REF/'runtime-library-hashes.txt')
    ex=soft.pop('/proc/1/exe')
    soft.update({'/candidate/libggml-cuda.so':p.DSO,'/proc/99/exe':ex,str(p.base.SERVER_IMPL):p.n9.hash_file(p.base.SERVER_IMPL)})
    (o/'runtime-library-hashes.txt').write_text(''.join(f'{v}  {k}\n' for k,v in soft.items()))
    (o/'supervisor.sha256').write_text(p.n9.hash_file(p.SUP)+'  /proc/1/exe\n')
    base={'child':99,'combined':0,'skip':0,'row0':0,'row1':0}
    write(w/'start.json',base)
    for phase in ('cold','warm'):
        t=p.read(o/(phase+'.json'))['timings'];calls=(t.get('draft_n') or 0)*40
        base['combined']+=calls;base['skip']+=calls;write(w/(phase+'.json'),base)
    result=p.collect(mode,o);write(o/'result.json',result)
    (o/'status.txt').write_text('runner_exit=0\ncleanup_exit=0\n')
    return o
with TemporaryDirectory(dir=p.HERE) as tmp:
    root=Path(tmp)
    a=stage(root,'serial',Path(os.environ["D1_SERIAL_REPLAY"]))
    b=stage(root,'native',p.REF)
    d=p.compare(a,b)
    assert d['measured']['warm']['native_text_matches_archived_n88'] and d['measured']['warm']['native_vs_n88_host_ms_per_round_saved']==0.
    orig=p.read(b/'witness/warm.json')
    for k in ['combined','skip','row0','row1','child']:
        bad=orig.copy();bad[k]+=1;write(b/'witness/warm.json',bad)
        try:p.counters('native',b,p.read(b/'result.json')['metrics'])
        except ValueError:pass
        else:raise ValueError('accepted altered '+k)
    write(b/'witness/warm.json',orig)
print('Historical 80K native and serial shapes, acceptance/parity, strict comparison and all damaged counter or child refusals passed. Candidate counts here are synthetic.')
