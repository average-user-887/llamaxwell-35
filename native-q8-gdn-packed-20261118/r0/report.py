import json
import re
import sys
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
archive=here.parent/'night-gdn-t2-q8-fast/receipts/nmmoon-n82-2k-20261118T034505Z-2895756-1198/b1'
layers=[i for i in range(40) if i%4!=3]
names=['t2.prefix.state','t2.badpair.state','t2.kept.state','t2.corrected.state','t2.badpair.logits','t2.corrected.logits']
sizes=[65864428]*4+[2*248320*4,248320*4]

def need(value,msg):
    if not value:
        raise ValueError(msg)

def keyed(path):
    a=[line.split() for line in path.read_text().splitlines()]
    need(all(len(v)==2 for v in a) and len(set(v[1] for v in a))==len(a),'invalid receipt keys')
    return {k:v for v,k in a}

def layout():
    permutation=[9*((d%128)//4)+(d//128)*4+d%4 if d<256 else 9*(d-256)+8 for d in range(288)]
    need(sorted(permutation)==list(range(288)),'packing is not bijective')
    for lane in range(32):
        indices=[lane*4+j for j in range(4)]+[128+lane*4+j for j in range(4)]+[256+lane]
        need([permutation[j] for j in indices]==list(range(9*lane,9*lane+9)), 'lane weight words differ')
    need(12352*1152*30==426885120,'unexpected total bytes')

def arm(root,name):
    p=root/name
    log=(p/'t2.log').read_text()
    q=name.startswith('b')
    need((p/'t2.rows').read_bytes()==(archive/'t2.rows').read_bytes(),'full row choices / forced rollback differ')
    need(keyed(p/'outputs.sha256')==keyed(archive/'outputs.sha256'),'state or logits differs from original Q8')
    need(keyed(p/'outputs.bytes')==dict(zip(names,map(str,sizes))),'truncated state/logits')
    required=['offloaded 42/42 layers to GPU','N82_ADMISSION original_blocks=3 candidate_blocks=3 grid=72 admitted=1',
              'Q8CLK_COHORT fallback0=0 fallback1=0 paired=40 skipped=40',
              'Q2_COHORT residency device=0 sm=24 active=2 regs=63 static=96 dynamic=39424 local=0 ready=1',
              'Q8PA_DSO '+(here/'candidate.sha256').read_text().split()[0]+'  /candidate/libggml-cuda.so']
    need(all(log.count(x)==1 for x in required),'GPU offload, Q8, cohort or library mismatch')
    need(re.search(r'^print_info: n_expert_used\s+= 8$',log,re.M),'eight experts')
    need(log.count('N34_CUDA_FORWARD symbol=_Z23ggml_cuda_occamy_gdn_t2')==1,'physical GDN DSO')
    admissions=re.findall(r'^Q8PA_ADMISSION grid=72 active=3 regs=(\d+) local=0$',log,re.M)
    slots=re.findall(r'^N82_EVENT specialization=(\d+) grid=72 slot=(\d+) tensor=blk\.(\d+)\.attn_qkv\.weight us=([\d.]+)$',log,re.M)
    need(len(slots)==30 and log.count('N82_EVENT ')==30,'wrong number of named CUDA events')
    need([(int(m),int(i),int(l)) for m,i,l,_ in slots]==[(3 if q else 2,i,l) for i,l in enumerate(layers)],'physical specialization/slot/layer')
    times=np.array([float(v) for *_,v in slots])
    need(np.isfinite(times).all() and ((times>0)&(times<20000)).all(),'bad event timing')
    copies=re.findall(r'^Q8PA_COPY tensor=blk\.(\d+)\.attn_qkv\.weight bytes=(\d+) copy_ms=([\d.]+) resident_bytes=(\d+)$',log,re.M)
    need(len(copies)==(30 if q else 0) and len(admissions)==int(q),'packed admission/copies')
    if q:
        need(int(admissions[0])<=80,'actual register count')
        need([(int(l),int(b),int(res)) for l,b,_,res in copies]==[(l,14229504,(i+1)*14229504) for i,l in enumerate(layers)],'packed order or size')
        need(all(float(ms)>0 for _,_,ms,_ in copies),'copy timing')
        need(log.rfind('Q8PA_COPY')<log.index('Q8PA_ADMISSION'),'copy took place inside timed q2')
    return {'specialization':3 if q else 2,'actual_rows':2,'experts':8,'paired_moe':40,'skipped_moe_callbacks':40,'row_fallbacks':0,
            'full_states_logits_choices_match':True,'kernel_event_us':times.tolist(),'mean_us':float(times.mean()),
            'pack_copy_ms_per_layer':[float(ms) for _,_,ms,_ in copies],'sum_pack_kernel_ms':sum(float(ms) for _,_,ms,_ in copies),
            'allocated_mirror_bytes':426885120 if q else 0,'q2_grid':72,'resident_ctas_per_sm':3}

def calculate(arms):
    deltas=[np.array(arms[a]['kernel_event_us'])-np.array(arms[b]['kernel_event_us']) for a,b in [('a1','b1'),('a2','b2')]]
    means=[float(d.mean()) for d in deltas]
    return {'mean_saving_us_per_layer':means,'sum_saving_ms_per_30':[float(d.sum()/1000) for d in deltas],
            'solo_both_pairs_ge28':min(means)>=28,'both_pairs_ge21_for_conditional_combination':min(means)>=21,
            'independent_R2_saving_required_for_conditional_gate_us_per_layer':23,
            'per_layer_pair_savings_us':[d.tolist() for d in deltas]}

if __name__=='__main__':
    layout()
    if sys.argv[1]=='--check':
        arms={k:{'kernel_event_us':[120. if k[0]=='a' else 99.]*30} for k in ['a1','b1','b2','a2']}
        result=calculate(arms)
        need(not result['solo_both_pairs_ge28'] and result['both_pairs_ge21_for_conditional_combination'],'conditional screen')
        arms['b2']['kernel_event_us']=[99.1]*30
        need(not calculate(arms)['both_pairs_ge21_for_conditional_combination'],'both pairs required')
        print('All 288 packed words and both order-reversed threshold boundaries verified.')
    else:
        root=Path(sys.argv[1]); arms={k:arm(root,k) for k in ['a1','b1','b2','a2']}
        print(json.dumps({'technical_pass':True,'layers':layers,'run_order':list(arms),'arms':arms,'timing':calculate(arms),
            'scope':'Four fresh processes, one two-row verification each on one real fixed 2K history. Weight copies completed during prefix, outside timed q2 and kernel event sums. Copy event time excludes allocation. Exact rollback state/logit/row parity checked locally.'},indent=2))
