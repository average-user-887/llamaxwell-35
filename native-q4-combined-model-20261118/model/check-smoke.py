"""Width-three archived smoke streams with explicit synthetic width-four route/count fields."""
import native_probe as n
import json,re,struct
from pathlib import Path
from tempfile import TemporaryDirectory
from observation import MAGIC
from trace_export import annotated

def main():
    old=n.C/'resume-q3-moe-native/receipts'
    d3=next((n.C/'resume-native-width/receipts').glob('depth3-*/private/server-start.log')).read_text()
    for mode in ('smoke-off','smoke-on'):
        src=next(old.glob(mode+'-*/private'))
        with TemporaryDirectory(dir=n.HERE) as tmp:
            p=Path(tmp)
            for f in ('admission.txt','quality-request.json','quality-score.json','quality-response.json'): (p/f).symlink_to(src/f)
            env,args=n.settings(mode,False)
            for f,rows in [('environment.args',env),('server.args',args),('pid1-server.args',['/candidate/llama-server',*args]),('pid1-flags.txt',env[1::2])]:
                (p/f).write_text('\n'.join(rows)+'\n')
            maps=n.original.mapped(n.REF);maps['/candidate/libggml-cuda.so']=n.Q4_CUDA
            maps[str(n.C/'night-cache-only/libs/libllama-server-impl.so')]=n.SERVER_IMPL;maps['/audit.so']=n.n9.hash_file(n.HERE/'audit.so')
            (p/'runtime-library-hashes.txt').write_text(''.join(f'{v}  {k}\n' for k,v in maps.items()))
            start,first,rest,full=n.log_phases(src)
            post=(src/'server-post-warm.log').read_text()
            before=(src/'server-pre-warm.log').read_text()
            second=post[len(before):]
            quality=full[len(post):]
            start=d3
            if mode=='smoke-on':start+='Q4_COHORT residency device=0 sm=24 active_per_sm=2 regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1\n'
            logs=[start];w=[0]*512;w[:3]=[MAGIC,1,1]
            (p/'counter-start.bin').write_bytes(struct.pack('<512Q',*w))
            for phase,log in [('cold',first),('warm',second)]:
                raw=n.n9.read_json(src/f'{phase}.json')
                log=re.sub(r'(accepted\s+\d+/\s*)2( draft tokens)',r'\g<1>3\2',log)
                offers=[int(b) for _,b,_ in n.original.n92.ACCEPT.findall(log)]
                raw['timings']['draft_n']=sum(offers)
                n.n9.write_json(p/f'{phase}.json',raw)
                k=offers.count(3);w[161]+=1;w[162]+=k;w[166]+=k
                if mode=='smoke-on':
                    w[5]+=40*k;w[6]=w[5]
                    for i in range(40):w[16+i]+=k;w[112+i]+=k
                (p/f'counter-{phase}.bin').write_bytes(struct.pack('<512Q',*w))
                logs.append(log)
            prefix=start
            for name,chunk in [('server-start.log',''),('server-pre-warm.log',logs[1]),('server-post-warm.log',logs[2]),('server.log',quality)]:
                prefix+=chunk;(p/name).write_text(prefix)
            result=n.collect(mode,False,p);assert result['semantic_score']['passed']
            assert any(r['event']=='cycle' and r['phase']=='quality' for r in annotated(mode,p))
    print('Two short report paths replayed with an actual archived arithmetic witness and synthetic depth-three routing fields.')
if __name__=='__main__':main()
