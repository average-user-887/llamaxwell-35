import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

LAYERS = [i for i in range(40) if i % 4 != 3]
NAMES = ['t2.prefix.state', 't2.badpair.state', 't2.kept.state', 't2.corrected.state', 't2.badpair.logits', 't2.corrected.logits']
SIZES = [65864428] * 4 + [2 * 248320 * 4, 248320 * 4]
REQUIRED_US_PER_LAYER = 819.12 / 30

def require(ok, message):
    if not ok:
        raise ValueError(message)

def lines_dict(path):
    return {line.split()[1]:line.split()[0] for line in path.read_text().splitlines()}

def read_arm(root, mode):
    rows = (root/mode/'t2.rows').read_text()
    log = (root/mode/'t2.log').read_text()
    require(len(re.findall(r'offloaded 42/42 layers to GPU', log))==1, 'physical offload')
    require('N34_START mode=t2 offload=42/42 rs=1 prompt=2000 ' in rows, 'wrong 2K fixture')
    require(rows.count('N34_CALL stage=badpair ')==1 and
            re.search(r'^N34_CALL stage=badpair at=2004 graphs=1 in=2 out=2 input0=\d+ input_last=\d+ ',rows,re.M),'actual q2 proof')
    require('N34_END mode=t2 t2_attempts=30 t2_launches=30 stock=0 stock_cache=0 wrong_shape=0 end=2009 ' in rows,'T2 dispatch counts')
    require('N34_RUNTIME llama=/candidate/libllama.so' in log and
            re.search(r'N34_CUDA_FORWARD symbol=_Z23ggml_cuda_occamy_gdn_t2\S* path=/candidate/libggml-cuda.so',log),'runtime origin')
    require(log.count(f'Q8CLK_BEGIN mode={mode} cc=52 grid=72 sm=24 standard_blocks=3 diagnostic_blocks=3')==1 and
            log.count(f'Q8CLK_END mode={mode} distinct_t2_layers=30')==1, 'specialization/occupancy')
    require(log.count('N82_ADMISSION original_blocks=3 candidate_blocks=3 grid=72 admitted=1')==1, 'Q8 specialization admission')
    require(log.count('Q8CLK_COHORT fallback0=0 fallback1=0 paired=40 skipped=40')==1, 'active MoE did not match the native profile')
    require(re.search(r'Q2_COHORT residency device=0 sm=24 active=2 regs=63 static=96 dynamic=39424 local=0 ready=1',log), 'actual MoE residency')
    require(re.search(r'Q8CLK_DSO [a-f0-9]{64} +/candidate/libggml-cuda.so',log), 'guest DSO hash unavailable')
    ev=re.findall(r'^Q8CLK_EVENT mode=(\w+) layer_slot=(\d+) tensor=blk\.(\d+)\.attn_qkv\.weight us=([\d.]+)$', log, re.M)
    require(len(ev)==30 and log.count('Q8CLK_EVENT ')==30,'incomplete or unrecognized event')
    require(all(a==mode and int(slot)==i and int(layer)==LAYERS[i] for i,(a,slot,layer,_) in enumerate(ev)),'layer identity/order')
    times=np.array([float(s) for _,_,_,s in ev])
    require(np.all(np.isfinite(times)) and np.all((times>0)&(times<20000)),'event span invalid')
    require(lines_dict(root/mode/'outputs.bytes')==dict(zip(NAMES,map(str,SIZES))),'full state/logit output sizes')
    hashes=lines_dict(root/mode/'outputs.sha256')
    require(set(hashes)==set(NAMES) and all(re.fullmatch('[0-9a-f]{64}',s) for s in hashes.values()),'output digest set')
    return times,rows,hashes

def fractions(raw):
    # One clock domain per CTA. Interpret all six stamps relative to that
    # CTA's first warp's entry and unwrap the 32-bit counter before extrema.
    offset=(raw.astype(np.int64)-raw[:,:,0:1,0:1].astype(np.int64)+2**31)%2**32-2**31
    require(np.all(np.diff(offset,axis=3)>0),'incomplete or nonmonotonic local clock sequence')
    require(np.all(np.abs(offset)<10**7),'implausible local tick span')
    start=np.min(offset[:,:,:,0],axis=2)
    finish=np.max(offset[:,:,:,5],axis=2)
    total=finish-start
    before=np.max(offset[:,:,:,1],axis=2)
    after=np.min(offset[:,:,:,2],axis=2)
    before_b=np.max(offset[:,:,:,3],axis=2)
    after_b=np.min(offset[:,:,:,4],axis=2)
    require(np.all((start<before)&(before<=after)&(after<before_b)&(before_b<=after_b)&(after_b<finish)), 'barrier ordering')
    # The slowest CTA's full lifetime is the reference for the grid span.
    # All durations are same-CTA subtractions before cross-CTA maxima.
    critical=np.max(total,axis=1)
    return {
        'through_last_A_arrival':np.max(before-start,axis=1)/critical,
        'through_last_A_release':np.max(after-start,axis=1)/critical,
        'through_last_B_arrival':np.max(before_b-start,axis=1)/critical,
        'through_last_B_release':np.max(after_b-start,axis=1)/critical,
        'A_compute':np.max(before-start,axis=1)/critical,
        'B_compute':np.max(before_b-after,axis=1)/critical,
        'C_compute':np.max(finish-after_b,axis=1)/critical,
        'A_barrier_span':np.max(after-before,axis=1)/critical,
        'B_barrier_span':np.max(after_b-before_b,axis=1)/critical,
    }

def describe(v):
    return {k:float(x) for k,x in zip(['min','p10','median','mean','p90','max'],
                  [v.min(),np.percentile(v,10),np.median(v),np.mean(v),np.percentile(v,90),v.max()])}

def main(root):
    control, cr, ch = read_arm(root,'control')
    clock, dr, dh = read_arm(root,'clock')
    raw=np.fromfile(root/'q8.clock.u32',dtype='<u4')
    require(raw.size==30*72*8*6,'incomplete clock readback')
    f=fractions(raw.reshape(30,72,8,6))
    package=Path(__file__).resolve().parent
    history=package.parent/'night-gdn-t2-q8-fast/receipts/nmmoon-n82-2k-20261118T034505Z-2895756-1198/b1'
    reference=lines_dict(history/'outputs.sha256')
    require(all('Q8CLK_DSO '+(package/'diagnostic.sha256').read_text().split()[0]+'  /candidate/libggml-cuda.so'
                in (root/mode/'t2.log').read_text() for mode in ('control','clock')), 'guest DSO digest mismatch')
    identity=ch==dh and cr==dr
    expected=ch==reference and cr==(history/'t2.rows').read_text()
    per_arm={k:{arm:describe(v*t) for arm,t in [('control',control),('clock',clock)]} for k,v in f.items()}
    mean_bias=float(clock.mean()/control.mean()-1)
    p90=float(np.percentile(np.abs(clock-control)/control,90))
    stable=abs(mean_bias)<=0.05 and p90<=0.10
    return {
        'technical_sample':True, 'physical_layers':LAYERS,
        'exact_paired_states_logits_rows':identity, 'exact_vs_archived_Q8_states_logits_rows':expected,
        'hashes_vs_archived_Q8':{n:ch[n]==reference[n] for n in NAMES},
        'control_event_us':describe(control),'clock_event_us':describe(clock),
        'mean_event_bias_fraction':mean_bias,'p90_paired_absolute_relative_bias':p90,
        'bias_within_predeclared_limits':bool(stable),
        'directional_phase_attribution_eligible':bool(identity and expected and stable),
        'phase_us_estimates':per_arm,
        'required_whole_GDN_saving_us_per_layer_if_sole_source':REQUIRED_US_PER_LAYER,
        'per_layer':[{'layer':layer,'control_event_us':float(control[i]),'clock_event_us':float(clock[i]),
                      'clock_fractions':{k:float(v[i]) for k,v in f.items()}} for i,layer in enumerate(LAYERS)],
        'scope':'One real fixed 2K prefix at 2004, 30 separate q2 GDN layers. Approximate ratios use same-CTA durations with each whole-kernel event. Different CTAs may dominate different phases; estimates are not additive or measured savings.'
    }

if __name__=='__main__':
    result=main(Path(sys.argv[1]))
    print(json.dumps(result,indent=2))
    if not result['exact_paired_states_logits_rows'] or not result['exact_vs_archived_Q8_states_logits_rows']:
        raise SystemExit(2)
