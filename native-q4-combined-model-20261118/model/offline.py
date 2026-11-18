"""Replay four long parser paths on recorded model streams with synthesized new audit telemetry."""
import json
from pathlib import Path
import re,struct
from tempfile import TemporaryDirectory
import native_probe as n
from observation import MAGIC
from unittest.mock import patch

def lines(p,rows): p.write_text("".join(str(x)+"\n" for x in rows))
def main():
    n.qualify();n.check_moe()
    serial=n.C/"resume-native-width/receipts/@PRIVATE_SERIAL_RUN_NAME@/private"
    native=next((n.C/"resume-native-width/receipts").glob("depth3-*/private"))
    op=(n.OPERATOR/"nu2-a.log").read_text()
    cohort='Q4_COHORT residency device=0 sm=24 active_per_sm=2 regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1\n'
    modes=['serial-on','depth3-off','depth3-attn','depth3-combined']
    assert n.settings('serial-on',False)[0]==n.settings('depth3-combined',False)[0]
    env=[n.settings(m,False)[0] for m in modes]
    assert sum(a!=b for a,b in zip(env[1],env[2]))==sum(a!=b for a,b in zip(env[2],env[3]))==1
    with TemporaryDirectory(dir=n.HERE) as tmp:
        roots=[]
        for mode in modes:
            out=Path(tmp)/mode/'private';out.mkdir(parents=True);roots.append(out.parent)
            safe=out.parent/'sanitized';safe.mkdir()
            src=serial if mode=='serial-on' else native
            for p in ['cold.json','warm.json','admission.txt']: (out/p).symlink_to(src/p)
            env,args=n.settings(mode,False)
            for name,rows in [('environment.args',env),('server.args',args),('pid1-server.args',['/candidate/llama-server',*args]),('pid1-flags.txt',env[1::2])]:
                lines(out/name,rows)
            maps=n.original.mapped(n.REF)
            maps['/candidate/libggml-cuda.so']=n.Q4_CUDA
            maps[str(n.C/'night-cache-only/libs/libllama-server-impl.so')]=n.SERVER_IMPL
            maps['/audit.so']=n.n9.hash_file(n.HERE/'audit.so')
            lines(out/'runtime-library-hashes.txt',(f'{v}  {k}' for k,v in maps.items()))
            start,cold,warm,_=n.log_phases(src)
            if mode=='serial-on':
                cold='\n'.join(x for x in cold.splitlines() if 'occamy FA mrow dispatch n_q=4' not in x)+'\n'
                warm='\n'.join(x for x in warm.splitlines() if 'occamy FA mrow dispatch n_q=4' not in x)+'\n'
            if mode in ('depth3-attn','depth3-combined'):
                cold='\n'.join(x for x in cold.splitlines() if 'occamy FA mrow dispatch n_q=4' not in x)+'\n'
                warm='\n'.join(x for x in warm.splitlines() if 'occamy FA mrow dispatch n_q=4' not in x)+'\n'+op
            if mode in ('serial-on','depth3-combined'): start+=cohort
            (out/'server-start.log').write_text(start)
            (out/'server-pre-warm.log').write_text(start+cold)
            (out/'server.log').write_text(start+cold+warm)
            words=[0]*512;words[:3]=[MAGIC,1,1]
            for phase,raw in [('start',''),('cold',cold),('warm',warm)]:
                if phase!='start':
                    k=sum(int(row[1])==3 for row in n.original.n92.ACCEPT.findall(raw))
                    words[161]+=1;words[162]+=max(k,1);words[165]+=1;words[166]+=k
                    if mode in ('serial-on','depth3-combined'):
                        for layer in range(40):
                            words[16+layer]+=1+k;words[64+layer]+=1;words[112+layer]+=k
                        words[5]+=40*(1+k);words[6]=words[5]
                (out/f'counter-{phase}.bin').write_bytes(struct.pack('<512Q',*words))
            r=n.collect(mode,False,out);n.n9.write_json(out/'result.json',r)
            n.n9.write_json(safe/'receipt.json',{'mode':mode,'status':{'technical_pass':True},'profile_instrumented':False})
        summary=n.compare(roots)
        assert set(summary['comparisons'])=={'attention_only','moe_over_attention','combined_over_off','strict_native_over_serial'}
        assert n.compare(roots[1:])['strict_warm_2x'] is None
        broken=roots[-1]/'private/counter-warm.bin';w=list(struct.unpack('<512Q',broken.read_bytes()));w[112]+=1;broken.write_bytes(struct.pack('<512Q',*w))
        try:n.collect(modes[-1],False,roots[-1]/'private')
        except ValueError as err: assert 'classification' in str(err)
        else: raise ValueError('misattributed row accepted')
    print('Four long arms replayed using archived true streams and synthetic new counters/route markers. Pairwise attribution, strict ratio and bad-position refusal verified.')
if __name__=='__main__':main()
