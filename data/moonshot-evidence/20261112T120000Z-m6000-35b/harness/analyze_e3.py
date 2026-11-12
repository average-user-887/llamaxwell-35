#!/usr/bin/env python3
# Recompute E3 tables from raw run dirs (events.json/result.json) + runs.jsonl.  usage: analyze_e3.py <evidence_dir> [label-filter]
import sys,json,os,statistics,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import metrics as MX
E=sys.argv[1]; T={1:12.71,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447}
rows=[json.loads(l) for l in open(E+'/runs.jsonl')]
R=[r for r in rows if r.get('workload_id','').startswith('e3-') and r.get('workload_id') in('e3-alone','e3-mixed') and not r.get('warmup') and r.get('status')=='PASS' and str(r.get('cycle_id','')).startswith('c')]
def metr(r):
    ev=json.load(open(E+'/runs/'+r['run_id']+'/events.json')); S=ev['streams']; d=S['deep']['ts_ns']
    if r['workload_id']=='e3-mixed':
        a=ev['arrival_ns']; f=S['u16k']['ts_ns'][0]; m=MX.deep_metrics(d,a,f); m['ttft_16k_s']=(f-S['u16k']['t_send_ns'])/1e9
        c=MX.window_stats({'deep':d,'u16k':S['u16k']['ts_ns']}); m['agg_tps_common']=c['agg_tps']; m['common_window_s']=c['window_ns']/1e9; m['deep_tps_common']=c['per_stream_tps']['deep']; m['u16k_tps_common']=c['per_stream_tps']['u16k']
    else: m=MX.deep_metrics(d)
    return m
by=collections.defaultdict(list)
for r in R: by[(r['workload_id'],r['variant_id'],r['block'])].append((r,metr(r)))
KEYS={'e3-alone':['tps_own_span','itl_p50_ms','itl_p95_ms','itl_max_ms'],'e3-mixed':['ttft_16k_s','tps_own_span','tps_during_prefill','deep_tokens_during_prefill','itl_p50_during_prefill_ms','itl_p95_during_prefill_ms','itl_max_during_prefill_ms','itl_max_ms','itl_p50_ms','itl_p95_ms','agg_tps_common','common_window_s','deep_tps_common','u16k_tps_common','prefill_window_ms']}
def bm(wl,arm,blk,k):
    v=[m[k] for _,m in by.get((wl,arm,blk),[]) if m.get(k) is not None]; return statistics.mean(v) if v else None
blocks=sorted({r['block'] for r in R})
out=['# E3 recomputed (analyze_e3.py)','', 'runs used: %d (PASS, non-warm, cycle c*); blocks: %s'%(len(R),blocks),'']
for wl in('e3-alone','e3-mixed'):
    out+=['## '+wl,'','| metric | arm | '+' | '.join('blk%s'%b for b in blocks)+' | mean | B vs A (B/A-1 %) | block-level 95% t-CI of diff % |','|---|---|'+'---|'*(len(blocks)+3)]
    for k in KEYS[wl]:
        pa={arm:[bm(wl,arm,b,k) for b in blocks] for arm in 'AB'}
        dif=[(y/x-1)*100 for x,y in zip(pa['A'],pa['B']) if x and y is not None]
        if len(dif)>1:
            mu=statistics.mean(dif); h=T.get(len(dif)-1,2)*statistics.stdev(dif)/len(dif)**.5; ci='%+.1f..%+.1f'%(mu-h,mu+h); dm='%+.2f'%mu
        else: ci=dm='n/a'
        for arm in 'AB':
            v=pa[arm]; vv=[x for x in v if x is not None]
            out.append('| %s | %s | %s | %s | %s | %s |'%(k,arm,' | '.join('%.3f'%x if x is not None else 'null' for x in v),'%.3f'%statistics.mean(vv) if vv else 'null',dm if arm=='B' else '',ci if arm=='B' else ''))
    out.append('')
out+=['Formulas: gain = B/A-1 (rates); period reduction = 1-pB/pA with p=1/rate. Block value = mean of the block\'s 2 positions per arm (ABBA). CI: t-interval over per-block B/A-1 (df=blocks-1).','']
open(E+'/E3/analysis_recomputed.md','w').write('\n'.join(out)); print('\n'.join(out))
