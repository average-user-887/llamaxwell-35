#!/usr/bin/env python3
# e4_analyze.py <armdir>: segments prof/prof.csv into 4 phases by idle gaps (>=3 s after model load), step marker = per-step logits DtoH memcpy
# (size 0.000925 at N1, 0.001850 at N2 in nvprof's Size column), writes analysis/{summary.csv,phaseN_kernels.csv,e1_engagement.csv}.
import csv,sys,re,json,statistics,collections,os
ad=sys.argv[1]; AO=ad+'/analysis'; os.makedirs(AO,exist_ok=True)
rows=list(csv.reader(open(ad+'/prof/prof.csv'))); h=[i for i,r in enumerate(rows) if r and r[0]=='Start'][0]; hd=rows[h]; ci={c:i for i,c in enumerate(hd)}
u=rows[h+1]; tf={'s':1,'ms':1e-3,'us':1e-6}[u[ci['Start']]]; df={'s':1e3,'ms':1,'us':1e-3,'ns':1e-6}[u[ci['Duration']]]
def short(n): return re.sub(r'\(.*','',n.replace('void ',''))[:110]
ev=sorted([(float(r[ci['Start']])*tf,float(r[ci['Duration']])*df,r[ci['Name']],r[ci['Size']]) for r in rows[h+2:] if len(r)==len(hd)])
# segments: first gap >10 s = model load; then gaps >=3 s
segs=[];cur=[ev[0]]; 
for a,b in zip(ev,ev[1:]):
    if b[0]-a[0]>=3.0: segs.append(cur);cur=[b]
    else: cur.append(b)
segs.append(cur)
print('segments',[(round(s[0][0],1),round(s[-1][0],1),len(s)) for s in segs])
tp=json.load(open(ad+'/plain/timings.json')); tr=json.load(open(ad+'/prof/timings.json'))
def srv(t,ph):
    x=t[ph]
    if 'prompt_ms' in x: return x['prompt_ms']/x['prompt_n']
    return statistics.mean(v['predicted_ms']/v['predicted_n'] for v in x.values())
PH=['1_prefill4k','2_n1_shallow','3_n2_shallow','4_n2_deep']
seg_for={}
if len(segs)>=4: seg_for={PH[0]:segs[-4],PH[1]:segs[-3],PH[2]:segs[-2],PH[3]:segs[-1]}
S=[]; E1=[]
for ph in PH:
    if ph not in seg_for: print('no segment for',ph); continue
    seg=seg_for[ph]
    if ph==PH[0]:
        end=seg[-1][0]; t0=end-tr[ph]['prompt_ms']/1000-0.1; seg=[e for e in seg if e[0]>=t0]
        win=[seg]; ntok=tr[ph]['prompt_n']; unit='prompt token'
        per=[seg[-1][0]-seg[0][0]]; kernels=[e for e in seg if not e[2].startswith('[')]
        busy=sum(e[1] for e in kernels); period=per[0]*1000; nsteps=1
        # per-"token" figures: window / prompt_n
        grp=[kernels]; scale=ntok
    else:
        want='0.000925' if ph==PH[1] else '0.001850'
        mk=[i for i,e in enumerate(seg) if e[2].startswith('[CUDA memcpy DtoH') and e[3]==want]
        wins=[seg[mk[j]+1:mk[j+1]+1] for j in range(2,len(mk)-1)]   # drop first 2 steps (prompt tail)
        pers=[(seg[mk[j+1]][0]-seg[mk[j]][0])*1000 for j in range(2,len(mk)-1)]
        grp=[[e for e in w if not e[2].startswith('[')] for w in wins]
        busys=[sum(e[1] for e in g) for g in grp]
        nsteps=len(grp); scale=nsteps; period=statistics.mean(pers); busy=statistics.mean(busys)
        print(ph,'markers',len(mk),'steps used',nsteps,'period sd %.3f'%statistics.pstdev(pers))
    ag=collections.OrderedDict()
    for g in grp:
        for e in g: ag.setdefault(short(e[2]),[]).append(e[1])
    tot=sum(sum(v) for v in ag.values())
    with open(f'{AO}/phase_{ph}_kernels.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['kernel','calls_total','total_ms','avg_us','launches_per_'+('step' if ph!=PH[0] else 'window'),'us_per_'+('step' if ph!=PH[0] else 'window'),'pct_busy'])
        for k,v in sorted(ag.items(),key=lambda x:-sum(x[1])):
            n=len(grp) if ph!=PH[0] else 1
            w.writerow([k,len(v),round(sum(v),3),round(1000*sum(v)/len(v),2),round(len(v)/n,2),round(1000*sum(v)/n,1),round(100*sum(v)/tot,2)])
    top=sorted(ag.items(),key=lambda x:-sum(x[1]))[:3]
    n=len(grp) if ph!=PH[0] else 1
    ptxt=lambda c:'%s(%.0fus/step)'%(c[0][:40],1000*sum(c[1])/n)
    S.append([ph,nsteps,round(period,3),round(busy,3),round(period-busy,3),round(100*(period-busy)/period,1),round(srv(tp,ph),4),round(srv(tr,ph),4),round(100*(srv(tr,ph)/srv(tp,ph)-1),2),' | '.join(ptxt(c) for c in top)])
    # E1
    if ph!=PH[0]:
        key={'moe_v2':'moe_v2','gdn_fused(n1)':'occamy_gdn::gdn_fused<','gdn_fused2':'gdn_fused2','fa_ar(FA_ADDR)':'ofa_decode_hs_ar_kernel','fa_hs':'ofa_decode_hs_kernel','fa_base':'ofa_decode_kernel','fa_combine':'ofa_combine_kernel','n2gather_copy':'gate_copy_sm52'}
        for lab,sub in key.items():
            c=sum(1 for g in grp for e in g if sub in e[2]); E1.append([ph,lab,c,round(c/n,2)])
with open(AO+'/summary.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['phase','steps','period_ms_nvprof','busy_ms','gap_ms','gap_pct','server_ms_per_tok_plain','server_ms_per_tok_nvprof','nvprof_overhead_pct','top3_kernels']); w.writerows(S)
with open(AO+'/e1_engagement.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['phase','kernel_class','count','per_step']); w.writerows(E1)
for r in S: print(r)
for r in E1: print(r)
